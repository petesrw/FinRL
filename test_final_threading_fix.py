#!/usr/bin/env python3
"""
🧪 Test Final Threading Fix
Quick test to verify threading configuration happens properly
"""

# Test the same pattern as our main script
import torch
import multiprocessing as mp

# Configure PyTorch threading immediately
try:
    torch.set_num_threads(max(16, mp.cpu_count()))
    torch.set_num_interop_threads(max(8, mp.cpu_count() // 2))
    print("✅ Threading configured successfully at startup")
    threading_configured = True
except Exception as e:
    print(f"❌ Threading configuration failed: {str(e)}")
    threading_configured = False

# Now do some PyTorch operations to trigger "parallel work"
print("🔄 Starting PyTorch operations...")

# Test GPU if available
if torch.cuda.is_available():
    print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
    device = torch.device('cuda')
    x = torch.randn(100, 100, device=device)
    y = torch.matmul(x, x)
    result = y.sum().item()
    print(f"   GPU computation result: {result:.2f}")
else:
    print("💻 Using CPU")
    x = torch.randn(100, 100)
    y = torch.matmul(x, x)
    result = y.sum().item()
    print(f"   CPU computation result: {result:.2f}")

# Now try to set threading again (should fail gracefully)
print("\n🧪 Testing redundant threading configuration...")
try:
    torch.set_num_threads(16)
    print("⚠️ Unexpected: Second threading call succeeded")
except RuntimeError as e:
    if "parallel work has started" in str(e):
        print("✅ Expected 'parallel work started' error handled gracefully")
    else:
        print(f"⚠️ Unexpected RuntimeError: {str(e)[:50]}...")

print("\n🎉 Threading test completed - fix should be working!")
print("💡 The main script should now run without threading warnings.")
