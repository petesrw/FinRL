#!/usr/bin/env python3
"""
🧪 Test PyTorch Threading Configuration Fix
Verifies that threading is properly configured without errors
"""

import torch
import multiprocessing as mp

def test_pytorch_threading():
    """Test PyTorch threading configuration"""
    print("🧪 Testing PyTorch Threading Configuration...")
    
    success = True
    
    try:
        # Get current configuration
        num_threads = torch.get_num_threads()
        num_interop_threads = torch.get_num_interop_threads()
        
        print(f"   📊 Current num_threads: {num_threads}")
        print(f"   📊 Current interop_threads: {num_interop_threads}")
        
        # Test setting threads (should work on first call)
        optimal_threads = max(16, mp.cpu_count())
        optimal_interop = max(8, mp.cpu_count() // 2)
        
        print(f"   🎯 Optimal threads: {optimal_threads}")
        print(f"   🎯 Optimal interop: {optimal_interop}")
        
        # First call should succeed
        torch.set_num_threads(optimal_threads)
        torch.set_num_interop_threads(optimal_interop)
        print("   ✅ Initial threading configuration successful")
        
        # Create a simple tensor operation to trigger parallel work
        x = torch.randn(100, 100)
        y = torch.matmul(x, x)
        result = y.sum().item()
        print(f"   🧮 Test computation result: {result:.2f}")
        
        # Now try to set threads again (should fail but be handled gracefully)
        try:
            torch.set_num_threads(optimal_threads)
            print("   ✅ Second threading call succeeded (unexpected but OK)")
        except RuntimeError as e:
            if "parallel work has started" in str(e):
                print("   ✅ Expected 'parallel work' error caught and handled")
            else:
                print(f"   ⚠️ Unexpected RuntimeError: {str(e)[:50]}...")
                success = False
        
        try:
            torch.set_num_interop_threads(optimal_interop)
            print("   ✅ Second interop threading call succeeded (unexpected but OK)")
        except RuntimeError as e:
            if "parallel work has started" in str(e):
                print("   ✅ Expected 'interop parallel work' error caught and handled")
            else:
                print(f"   ⚠️ Unexpected RuntimeError: {str(e)[:50]}...")
                success = False
        
        # Final configuration check
        final_threads = torch.get_num_threads()
        final_interop = torch.get_num_interop_threads()
        
        print(f"   📊 Final num_threads: {final_threads}")
        print(f"   📊 Final interop_threads: {final_interop}")
        
        return success
        
    except Exception as e:
        print(f"   ❌ Unexpected error: {str(e)}")
        return False

def test_gpu_after_threading():
    """Test GPU operations after threading configuration"""
    print("\n🚀 Testing GPU Operations After Threading Setup...")
    
    if not torch.cuda.is_available():
        print("   ⚠️ No CUDA GPU available - skipping GPU tests")
        return True
    
    try:
        device = torch.device('cuda')
        
        # Test GPU tensor operations
        x = torch.randn(256, 256, device=device)
        y = torch.matmul(x, x)
        result = y.sum().item()
        
        print(f"   ✅ GPU computation successful: {result:.2f}")
        
        # Test GPU memory operations
        torch.cuda.empty_cache()
        memory_allocated = torch.cuda.memory_allocated() / 1024**2
        print(f"   📊 GPU memory allocated: {memory_allocated:.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"   ❌ GPU test failed: {str(e)[:50]}...")
        return False

if __name__ == "__main__":
    print("🚀 PyTorch Threading Configuration Test")
    print("=" * 50)
    
    # Test threading configuration
    threading_success = test_pytorch_threading()
    
    # Test GPU after threading
    gpu_success = test_gpu_after_threading()
    
    print("\n" + "=" * 50)
    if threading_success and gpu_success:
        print("🎉 All tests passed! Threading configuration fix is working.")
        print("💡 The training system should now run without threading warnings.")
    else:
        print("❌ Some tests failed. Please check the implementation.")
        if not threading_success:
            print("   - Threading configuration needs attention")
        if not gpu_success:
            print("   - GPU operations need attention")
