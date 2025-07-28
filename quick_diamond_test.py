#!/usr/bin/env python3
"""Quick test for DIAMOND model compatibility"""

import sys
sys.path.append('.')

try:
    from forex_system_with_config import ConfigurableForexBot
    
    print("🧪 Testing DIAMOND Model Compatibility")
    print("=" * 40)
    
    # Initialize bot
    bot = ConfigurableForexBot()
    
    # Load DIAMOND model
    model_path = "models/diamond/eurusd_diamond_score100_attempt1_20250728_215948.zip"
    print(f"📂 Loading model: {model_path}")
    
    if bot.load_model(model_path):
        print("✅ Model loaded successfully!")
        
        if hasattr(bot, 'expected_obs_shape'):
            print(f"✅ Expected observation shape: {bot.expected_obs_shape}")
        else:
            print("⚠️  No expected observation shape detected")
        
        # Create environment
        print("\n🏗️  Creating environment...")
        env = bot.create_training_environment()
        
        if env:
            print("✅ Environment created!")
            print(f"📐 Environment observation space: {env.observation_space.shape}")
            
            # Test reset
            print("\n🔄 Testing environment reset...")
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            
            print(f"✅ Reset successful! Observation shape: {obs.shape}")
            
            # Verify compatibility
            if hasattr(bot, 'expected_obs_shape'):
                if obs.shape == bot.expected_obs_shape:
                    print("🎉 PERFECT MATCH! Model and environment are compatible!")
                    print("✅ READY FOR LIVE TRADING!")
                else:
                    print(f"❌ Shape mismatch: model expects {bot.expected_obs_shape}, got {obs.shape}")
            
        else:
            print("❌ Failed to create environment")
    else:
        print("❌ Failed to load model")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
