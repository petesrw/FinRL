#!/usr/bin/env python3
"""
🚀 Final Threading Fix Test
Quick test to verify threading configuration is working correctly
"""

# Test exactly like the main script
print("🧪 Testing final threading configuration fix...")

# Configure PyTorch threading immediately (like main script)
import torch
import multiprocessing as mp

_threading_configured = False
try:
    torch.set_num_threads(max(16, mp.cpu_count()))
    torch.set_num_interop_threads(max(8, mp.cpu_count() // 2))
    _threading_configured = True
    print("✅ PyTorch threading configured at startup")
except Exception as e:
    print(f"❌ Threading setup failed: {str(e)}")

# Now do some PyTorch operations
print("🔄 Starting PyTorch operations...")

# Test GPU if available
if torch.cuda.is_available():
    device = torch.device('cuda')
    x = torch.randn(100, 100, device=device)
    y = torch.matmul(x, x)
    print(f"✅ GPU computation successful: {y.sum().item():.2f}")
else:
    x = torch.randn(100, 100)  
    y = torch.matmul(x, x)
    print(f"✅ CPU computation successful: {y.sum().item():.2f}")

# Test function-level threading (like the main script functions)
def test_function():
    global _threading_configured
    if not _threading_configured:
        try:
            torch.set_num_threads(16)
            torch.set_num_interop_threads(8)
            print("⚠️ Unexpected: Function-level threading succeeded")
        except RuntimeError as e:
            if "parallel work has started" in str(e):
                print("✅ Expected: Threading error handled correctly")
            else:
                print(f"❓ Unexpected error: {str(e)[:50]}...")
    else:
        print("✅ Function correctly skipped threading (already configured)")

test_function()

print("\n🎉 Threading test completed!")
print("💡 Main script should now run without threading warnings.")
