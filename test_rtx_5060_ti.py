#!/usr/bin/env python3
"""
Test RTX 5060 Ti with sm_120 architecture support
"""

import torch
import os

def test_rtx_5060_ti_optimization():
    print("🔥 RTX 5060 Ti sm_120 Architecture Test")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    # Get GPU info
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    gpu_capability = torch.cuda.get_device_capability(0)
    
    print(f"🚀 GPU: {gpu_name}")
    print(f"💾 Memory: {gpu_memory:.1f} GB")
    print(f"🎯 Compute Capability: {gpu_capability[0]}.{gpu_capability[1]}")
    
    # Check architecture
    major, minor = gpu_capability
    if major >= 12:
        arch_name = "sm_120"
        arch_list = "12.0"
        print(f"✅ Blackwell architecture detected: {arch_name}")
    elif major >= 9:
        arch_name = "sm_90"
        arch_list = "9.0"
        print(f"✅ Ada Lovelace architecture detected: {arch_name}")
    else:
        arch_name = f"sm_{major}{minor}"
        arch_list = f"{major}.{minor}"
        print(f"⚠️ Older architecture: {arch_name}")
    
    # Set optimal environment
    os.environ['TORCH_CUDA_ARCH_LIST'] = arch_list
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    
    print(f"🎯 Architecture configured: {arch_name}")
    print(f"🔧 TORCH_CUDA_ARCH_LIST: {arch_list}")
    
    # Test GPU functionality
    try:
        print("\n🧪 GPU Performance Test:")
        
        # Small test
        print("   Testing small matrix operations...")
        x = torch.randn(256, 256, device=device, dtype=torch.float32)
        y = torch.matmul(x, x.T)
        result = y.sum().item()
        print(f"   ✅ Small test passed: {result:.2f}")
        
        # Medium test
        print("   Testing medium matrix operations...")
        x = torch.randn(1024, 1024, device=device, dtype=torch.float32)
        y = torch.matmul(x, x.T)
        result = y.sum().item()
        print(f"   ✅ Medium test passed: {result:.2f}")
        
        # Large test (if enough memory)
        if gpu_memory >= 12:
            print("   Testing large matrix operations...")
            x = torch.randn(4096, 4096, device=device, dtype=torch.float32)
            y = torch.matmul(x, x.T)
            result = y.sum().item()
            print(f"   ✅ Large test passed: {result:.2f}")
        
        # Test FP16 if supported
        if major >= 9:  # Ada Lovelace or newer
            print("   Testing FP16 precision...")
            x = torch.randn(2048, 2048, device=device, dtype=torch.float16)
            y = torch.matmul(x, x.T)
            result = y.sum().item()
            print(f"   ✅ FP16 test passed: {result:.2f}")
        
        # Memory utilization test
        memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
        memory_cached = torch.cuda.memory_reserved(0) / 1024**3
        
        print(f"\n📊 Memory Usage:")
        print(f"   Allocated: {memory_allocated:.2f} GB")
        print(f"   Cached: {memory_cached:.2f} GB")
        print(f"   Available: {gpu_memory - memory_cached:.2f} GB")
        print(f"   Utilization: {(memory_cached/gpu_memory)*100:.1f}%")
        
        # Clean up
        del x, y
        torch.cuda.empty_cache()
        
        print(f"\n🎉 RTX 5060 Ti {arch_name} optimization test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_rtx_5060_ti_optimization()
    if success:
        print("\n✅ Ready for Enhanced Active Trading System!")
    else:
        print("\n❌ GPU optimization failed - check configuration")
