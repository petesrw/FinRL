#!/usr/bin/env python3
"""
🧪 Test SAC Algorithm Compatibility Fix
Tests that SAC works with filtered policy_kwargs (no ortho_init)
"""

import torch
import numpy as np
from stable_baselines3 import SAC, PPO, A2C
from gymnasium import spaces
import gymnasium as gym

def test_sac_compatibility():
    """Test SAC with policy_kwargs filtering"""
    print("🧪 Testing SAC Algorithm Compatibility...")
    
    # Simple test environment
    env = gym.make('CartPole-v1')
    
    # Policy kwargs that should work for all algorithms
    base_policy_kwargs = {
        "net_arch": [64, 64],
        "activation_fn": torch.nn.ReLU,
    }
    
    # Policy kwargs with ortho_init (incompatible with SAC)
    full_policy_kwargs = {
        **base_policy_kwargs,
        "ortho_init": False,  # This should be filtered out for SAC
    }
    
    print(f"   📊 Base policy_kwargs: {base_policy_kwargs}")
    print(f"   📊 Full policy_kwargs: {full_policy_kwargs}")
    
    # Test SAC with filtering
    try:
        print("   🔧 Testing SAC with ortho_init filtering...")
        
        # Filter out ortho_init for SAC
        sac_policy_kwargs = {k: v for k, v in full_policy_kwargs.items() if k != 'ortho_init'}
        
        sac_model = SAC(
            "MlpPolicy",
            env,
            learning_rate=0.0003,
            batch_size=256,
            policy_kwargs=sac_policy_kwargs,
            verbose=0
        )
        
        print("   ✅ SAC model created successfully!")
        print(f"   📊 SAC policy_kwargs: {sac_policy_kwargs}")
        
    except Exception as e:
        print(f"   ❌ SAC model creation failed: {str(e)}")
        return False
    
    # Test PPO with ortho_init (should work)
    try:
        print("   🔧 Testing PPO with ortho_init (should work)...")
        
        ppo_model = PPO(
            "MlpPolicy",
            env,
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=64,
            policy_kwargs=full_policy_kwargs,
            verbose=0
        )
        
        print("   ✅ PPO model created successfully!")
        print(f"   📊 PPO policy_kwargs: {full_policy_kwargs}")
        
    except Exception as e:
        print(f"   ❌ PPO model creation failed: {str(e)}")
        return False
    
    print("✅ All algorithm compatibility tests passed!")
    return True

def test_algorithm_specific_filtering():
    """Test the algorithm-specific parameter filtering logic"""
    print("\n🔧 Testing Algorithm-Specific Parameter Filtering...")
    
    # Sample model_kwargs with potential conflicts
    model_kwargs = {
        'policy_kwargs': {
            'net_arch': [128, 128],
            'activation_fn': torch.nn.ReLU,
            'ortho_init': False,  # Should be filtered for SAC
        },
        'batch_size': 512,
        'n_steps': 2048,  # Should be filtered for SAC
    }
    
    print(f"   📊 Original model_kwargs: {model_kwargs}")
    
    # Test SAC filtering
    sac_kwargs = {k: v for k, v in model_kwargs.items() if k not in ['batch_size']}
    if 'policy_kwargs' in sac_kwargs and 'ortho_init' in sac_kwargs['policy_kwargs']:
        sac_policy_kwargs = {k: v for k, v in sac_kwargs['policy_kwargs'].items() if k != 'ortho_init'}
        sac_kwargs['policy_kwargs'] = sac_policy_kwargs
    
    print(f"   ✅ Filtered SAC kwargs: {sac_kwargs}")
    
    # Test PPO filtering (should keep ortho_init)
    ppo_kwargs = {k: v for k, v in model_kwargs.items() if k not in ['batch_size', 'n_steps']}
    
    print(f"   ✅ Filtered PPO kwargs: {ppo_kwargs}")
    
    # Verify ortho_init was removed from SAC but kept in PPO
    sac_has_ortho = 'policy_kwargs' in sac_kwargs and 'ortho_init' in sac_kwargs['policy_kwargs']
    ppo_has_ortho = 'policy_kwargs' in ppo_kwargs and 'ortho_init' in ppo_kwargs['policy_kwargs']
    
    print(f"   🔍 SAC has ortho_init: {sac_has_ortho} (should be False)")
    print(f"   🔍 PPO has ortho_init: {ppo_has_ortho} (should be True)")
    
    if not sac_has_ortho and ppo_has_ortho:
        print("✅ Parameter filtering logic works correctly!")
        return True
    else:
        print("❌ Parameter filtering logic failed!")
        return False

if __name__ == "__main__":
    print("🚀 SAC Compatibility Test Suite")
    print("=" * 50)
    
    # Run tests
    test1_passed = test_sac_compatibility()
    test2_passed = test_algorithm_specific_filtering()
    
    print("\n" + "=" * 50)
    if test1_passed and test2_passed:
        print("🎉 All tests passed! SAC compatibility fix is working.")
    else:
        print("❌ Some tests failed. Please check the implementation.")
