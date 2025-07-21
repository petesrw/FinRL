#!/usr/bin/env python3
"""
GPU และ CUDA Configuration Checker สำหรับ RTX 5060 Ti
"""

import torch
import sys

def main():
    print("=" * 50)
    print("🔥 GPU & CUDA Configuration Check")
    print("=" * 50)
    
    # Basic PyTorch info
    print(f"📦 PyTorch version: {torch.__version__}")
    print(f"🐍 Python version: {sys.version}")
    
    # CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"⚡ CUDA available: {cuda_available}")
    
    if cuda_available:
        print(f"🔧 CUDA version (PyTorch): {torch.version.cuda}")
        print(f"🏗️ CUDNN version: {torch.backends.cudnn.version()}")
        
        # GPU info
        gpu_count = torch.cuda.device_count()
        print(f"🎮 GPU count: {gpu_count}")
        
        for i in range(gpu_count):
            print(f"\n🔥 GPU {i}:")
            print(f"   📛 Name: {torch.cuda.get_device_name(i)}")
            
            # Memory info
            memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
            memory_cached = torch.cuda.memory_reserved(i) / 1024**3
            
            print(f"   💾 Total Memory: {memory_total:.1f} GB")
            print(f"   🔄 Allocated: {memory_allocated:.1f} GB")
            print(f"   📦 Cached: {memory_cached:.1f} GB")
            print(f"   💚 Available: {memory_total - memory_cached:.1f} GB")
            
            # Compute capability
            capability = torch.cuda.get_device_capability(i)
            print(f"   🎯 Compute capability: {capability[0]}.{capability[1]}")
            
            # RTX 5060 Ti specific checks
            if "RTX 5060" in torch.cuda.get_device_name(i):
                print(f"   🚀 RTX 5060 Ti detected!")
                
                # Check if sm_120 is supported
                major, minor = capability
                if major >= 12:  # Blackwell architecture
                    print(f"   ✅ sm_{major}{minor} supported (Blackwell architecture)")
                    print(f"   🎁 Advanced features: Tensor Cores 5th gen, FP16/BF16 support")
                elif major >= 9:  # Ada Lovelace or newer
                    print(f"   ✅ sm_{major}{minor} supported (Ada Lovelace+)")
                elif major >= 8:  # Ampere
                    print(f"   ⚠️ sm_{major}{minor} (Ampere) - older architecture")
                else:
                    print(f"   ❌ sm_{major}{minor} - very old architecture")
        
        # Test GPU functionality
        print("\n🧪 GPU Functionality Test:")
        try:
            # Simple tensor operations
            x = torch.randn(1000, 1000, device='cuda')
            y = torch.randn(1000, 1000, device='cuda')
            z = torch.mm(x, y)
            print("   ✅ Matrix multiplication: PASSED")
            
            # Memory cleanup
            del x, y, z
            torch.cuda.empty_cache()
            print("   ✅ Memory management: PASSED")
            
        except Exception as e:
            print(f"   ❌ GPU test failed: {e}")
    
    else:
        print("❌ CUDA not available")
        print("💡 Possible solutions:")
        print("   - Install CUDA-enabled PyTorch")
        print("   - Check NVIDIA drivers")
        print("   - Verify GPU compatibility")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    main()
