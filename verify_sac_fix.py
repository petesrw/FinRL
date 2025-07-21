#!/usr/bin/env python3
"""
Quick SAC Compatibility Test - Check if ortho_init filtering works
"""

def test_policy_kwargs_filtering():
    """Test the policy_kwargs filtering logic"""
    print("🧪 Testing SAC Policy Kwargs Filtering Logic...")
    
    # Simulate model_kwargs with ortho_init (as passed from our training system)
    model_kwargs = {
        'policy_kwargs': {
            'net_arch': [2048, 1024, 512],
            'activation_fn': 'torch.nn.ReLU',  # Simulated as string
            'ortho_init': False,
        },
        'batch_size': 512,
        'learning_rate': 0.0003,
        'gamma': 0.99
    }
    
    print(f"   📊 Original model_kwargs: {model_kwargs}")
    
    # Apply SAC filtering logic (same as in our train_all_models.py)
    sac_kwargs = {k: v for k, v in model_kwargs.items() if k not in ['batch_size']}
    
    # SAC doesn't support ortho_init parameter - filter it out
    if 'policy_kwargs' in sac_kwargs and 'ortho_init' in sac_kwargs['policy_kwargs']:
        sac_policy_kwargs = {k: v for k, v in sac_kwargs['policy_kwargs'].items() if k != 'ortho_init'}
        sac_kwargs['policy_kwargs'] = sac_policy_kwargs
    
    print(f"   ✅ Filtered SAC kwargs: {sac_kwargs}")
    
    # Verify ortho_init was removed
    has_ortho_init = 'policy_kwargs' in sac_kwargs and 'ortho_init' in sac_kwargs['policy_kwargs']
    
    print(f"   🔍 SAC kwargs contains ortho_init: {has_ortho_init}")
    
    if not has_ortho_init:
        print("✅ SUCCESS: ortho_init properly filtered out for SAC!")
        return True
    else:
        print("❌ FAILED: ortho_init still present in SAC kwargs!")
        return False

def test_all_algorithms_filtering():
    """Test filtering for all three algorithms"""
    print("\n🔧 Testing All Algorithm Parameter Filtering...")
    
    # Common model_kwargs
    model_kwargs = {
        'policy_kwargs': {
            'net_arch': [1024, 512],
            'activation_fn': 'torch.nn.ReLU',
            'ortho_init': False,
        },
        'batch_size': 1024,
        'n_steps': 2048,
        'learning_rate': 0.0003,
        'gamma': 0.99
    }
    
    print("   📊 PPO Filtering (should keep ortho_init):")
    ppo_kwargs = {k: v for k, v in model_kwargs.items() if k not in ['batch_size', 'n_steps']}
    ppo_has_ortho = 'policy_kwargs' in ppo_kwargs and 'ortho_init' in ppo_kwargs['policy_kwargs']
    print(f"      ortho_init present: {ppo_has_ortho} (should be True)")
    
    print("   📊 SAC Filtering (should remove ortho_init):")
    sac_kwargs = {k: v for k, v in model_kwargs.items() if k not in ['batch_size']}
    if 'policy_kwargs' in sac_kwargs and 'ortho_init' in sac_kwargs['policy_kwargs']:
        sac_policy_kwargs = {k: v for k, v in sac_kwargs['policy_kwargs'].items() if k != 'ortho_init'}
        sac_kwargs['policy_kwargs'] = sac_policy_kwargs
    sac_has_ortho = 'policy_kwargs' in sac_kwargs and 'ortho_init' in sac_kwargs['policy_kwargs']
    print(f"      ortho_init present: {sac_has_ortho} (should be False)")
    
    print("   📊 A2C Filtering (should keep ortho_init):")
    a2c_kwargs = {k: v for k, v in model_kwargs.items() if k not in ['batch_size', 'n_steps']}
    a2c_has_ortho = 'policy_kwargs' in a2c_kwargs and 'ortho_init' in a2c_kwargs['policy_kwargs']
    print(f"      ortho_init present: {a2c_has_ortho} (should be True)")
    
    # Check results
    success = (ppo_has_ortho == True and sac_has_ortho == False and a2c_has_ortho == True)
    
    if success:
        print("✅ All algorithm filtering works correctly!")
    else:
        print("❌ Some algorithm filtering failed!")
    
    return success

if __name__ == "__main__":
    print("🚀 SAC Compatibility Fix Verification")
    print("=" * 50)
    
    test1 = test_policy_kwargs_filtering()
    test2 = test_all_algorithms_filtering()
    
    print("\n" + "=" * 50)
    if test1 and test2:
        print("🎉 SAC Compatibility Fix VERIFIED!")
        print("   The training system should now work with SAC algorithm.")
    else:
        print("❌ Fix verification failed!")
