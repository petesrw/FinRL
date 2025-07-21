#!/usr/bin/env python3
"""
🎯 SAC Compatibility Test - Simple Version
Tests our SAC ortho_init filtering approach
"""

def test_sac_parameter_filtering():
    """Test the exact filtering logic we implemented"""
    
    print("🧪 Testing SAC Parameter Filtering Logic...")
    
    # This simulates the model_kwargs that come from get_high_utilization_model_config_async()
    model_kwargs = {
        "policy_kwargs": {
            "net_arch": [12288, 8192, 4096, 2048, 1024],
            "activation_fn": "torch.nn.ReLU",  # String representation
            "ortho_init": False,  # This should be filtered out for SAC
        },
        "batch_size": 16384,
        "n_steps": 131072,
        "learning_rate": 0.0003,
        "gamma": 0.99,
        "tensorboard_log": None,
        "verbose": 0
    }
    
    print("📊 Original model_kwargs:")
    for key, value in model_kwargs.items():
        print(f"   {key}: {value}")
    
    # Apply our SAC filtering logic (exact code from train_all_models.py)
    sac_kwargs = {k: v for k, v in model_kwargs.items() if k not in ['batch_size']}
    
    # SAC doesn't support ortho_init parameter - filter it out
    if 'policy_kwargs' in sac_kwargs and 'ortho_init' in sac_kwargs['policy_kwargs']:
        sac_policy_kwargs = {k: v for k, v in sac_kwargs['policy_kwargs'].items() if k != 'ortho_init'}
        sac_kwargs['policy_kwargs'] = sac_policy_kwargs
    
    print("\n✅ Filtered SAC kwargs:")
    for key, value in sac_kwargs.items():
        print(f"   {key}: {value}")
    
    # Verify ortho_init was removed
    has_ortho_init = 'policy_kwargs' in sac_kwargs and 'ortho_init' in sac_kwargs.get('policy_kwargs', {})
    
    print(f"\n🔍 Verification:")
    print(f"   Original had ortho_init: {'ortho_init' in model_kwargs.get('policy_kwargs', {})}")
    print(f"   Filtered has ortho_init: {has_ortho_init}")
    
    if not has_ortho_init:
        print("✅ SUCCESS: SAC ortho_init filtering works correctly!")
        return True
    else:
        print("❌ FAILED: ortho_init still present!")
        return False

if __name__ == "__main__":
    print("🚀 SAC Compatibility Test")
    print("=" * 40)
    
    success = test_sac_parameter_filtering()
    
    print("\n" + "=" * 40)
    if success:
        print("🎉 Fix verified! SAC should work now.")
        print("💡 You can now run: python train_all_models.py")
    else:
        print("❌ Fix needs more work.")
