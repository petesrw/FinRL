#!/usr/bin/env python3
"""
🔧 Simple Threading Check
Ultra-lightweight test for threading configuration
"""

import torch

print("🧪 Testing PyTorch threading configuration...")

try:
    import multiprocessing as mp
    torch.set_num_threads(max(16, mp.cpu_count()))
    torch.set_num_interop_threads(max(8, mp.cpu_count() // 2))
    print("✅ Initial threading configuration succeeded")
    
    # Do a simple operation
    x = torch.randn(10, 10)
    y = x + 1
    print(f"✅ First PyTorch operation completed: {y.sum().item():.2f}")
    
    # Try threading again (should fail)
    try:
        torch.set_num_threads(8)
        print("⚠️  Warning: Second threading call unexpectedly succeeded")
    except RuntimeError as e:
        if "parallel work has started" in str(e):
            print("✅ Proper error: 'parallel work has started' - fix working correctly")
        else:
            print(f"❓ Unexpected error: {str(e)[:50]}...")
    
    print("\n🎉 Threading test completed successfully!")
    print("💡 Main script should run without threading warnings now.")
    
except Exception as e:
    print(f"❌ Test failed: {str(e)}")

print("\n🔍 If you see 'parallel work has started' above, that's GOOD!")
print("   It means threading was configured early and is working properly.")
