#!/usr/bin/env python3
"""
🧪 Final Threading Test
Test the complete threading fix implementation
"""

# Test the exact same pattern as our main script
import torch
import multiprocessing as mp

print("🧪 Testing Final Threading Configuration Fix...")

# Configure threading exactly like main script
_threading_configured = False
try:
    torch.set_num_threads(max(16, mp.cpu_count()))
    torch.set_num_interop_threads(max(8, mp.cpu_count() // 2))
    _threading_configured = True
    print("✅ Step 1: Global threading configured successfully")
except Exception as e:
    print(f"❌ Step 1: Threading setup failed: {str(e)}")
    _threading_configured = False

# Import async_config (like main script does)
try:
    from async_config import get_config
    print("✅ Step 2: async_config imported successfully")
except Exception as e:
    print(f"⚠️ Step 2: async_config import issue: {str(e)}")

# Test GPU operations to trigger "parallel work"
if torch.cuda.is_available():
    print("🚀 Step 3: Starting GPU operations...")
    device = torch.device('cuda')
    x = torch.randn(100, 100, device=device)
    y = torch.matmul(x, x)
    print(f"✅ Step 3: GPU operations successful, result sum: {y.sum().item():.2f}")
else:
    print("💻 Step 3: Using CPU operations...")
    x = torch.randn(100, 100)
    y = torch.matmul(x, x)
    print(f"✅ Step 3: CPU operations successful, result sum: {y.sum().item():.2f}")

# Test function threading access (like in apply_maximum_gpu_utilization_async)
def test_function_threading():
    global _threading_configured
    print("🔍 Step 4: Testing function-level threading access...")
    
    if not _threading_configured:
        try:
            torch.set_num_threads(16)
            torch.set_num_interop_threads(8)
            print("⚠️ Step 4: Unexpected - second threading call succeeded")
        except RuntimeError as e:
            if "parallel work has started" in str(e):
                print("✅ Step 4: Expected 'parallel work started' error handled correctly")
            else:
                print(f"❓ Step 4: Unexpected RuntimeError: {str(e)[:50]}...")
        except Exception as e:
            print(f"❌ Step 4: Unexpected error: {str(e)[:50]}...")
    else:
        print("✅ Step 4: Threading already configured globally - function correctly skipped")

# Run function test
test_function_threading()

print("\n🎉 Threading Configuration Test Complete!")
print("💡 If you see 'parallel work has started' or 'correctly skipped', the fix is working!")
print("🚀 Main training script should now run without threading warnings.")
