#!/usr/bin/env python3
"""
🚀 Async Training Configuration
Advanced settings for async multi-model training
"""

import torch
import multiprocessing as mp
import os

# Skip threading configuration - handled globally by main script
# (Threading configuration moved to train_all_models.py startup)

class AsyncTrainingConfig:
    """Configuration class for async training"""
    
    def __init__(self):
        self.detect_hardware()
        self.set_optimal_config()
    
    def detect_hardware(self):
        """Detect and analyze hardware capabilities"""
        # CPU Detection
        self.cpu_cores = mp.cpu_count()
        self.cpu_threads = self.cpu_cores * 2  # Assuming hyperthreading
        
        # GPU Detection
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            self.gpu_count = torch.cuda.device_count()
            self.gpu_name = torch.cuda.get_device_name(0)
            self.gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            self.gpu_compute_capability = torch.cuda.get_device_capability(0)
        else:
            self.gpu_count = 0
            self.gpu_name = "No GPU"
            self.gpu_memory_gb = 0
            self.gpu_compute_capability = (0, 0)
        
        # System Memory (approximate)
        try:
            import psutil
            self.system_memory_gb = psutil.virtual_memory().total / 1024**3
        except ImportError:
            self.system_memory_gb = 16  # Default assumption
    
    def set_optimal_config(self):
        """Set optimal configuration based on hardware"""
        
        # RTX 5060 TI Specific Optimization - ENHANCED 4-MODEL CONCURRENT TRAINING
        if "RTX 5060" in self.gpu_name or "RTX 50" in self.gpu_name:
            self.max_concurrent_models = 4  # เพิ่มจาก 2 เป็น 4 เพื่อเทรน 4 models พร้อมกัน
            self.memory_per_model = 0.25  # ลดจาก 0.45 เป็น 0.25 (25% per model × 4 = 100%)
            self.batch_size_multiplier = 32  # คงเดิม
            self.timesteps_multiplier = 10   # คงเดิม
            self.neural_network_size = "ultra"  # คงเดิม
            self.gpu_memory_fraction = 0.95  # คงเดิม
            
        # High-end GPUs (16GB+) - OPTIMIZED FOR RTX 5060 TI with sm_120 (Blackwell)  
        elif self.gpu_memory_gb >= 16:
            self.max_concurrent_models = 2  # ลดจาก 4 เป็น 2 เพื่อ memory balance
            self.memory_per_model = 0.45  # ลดจาก 0.80 เป็น 0.45 (45% per model × 2 = 90%)
            self.batch_size_multiplier = 32  # เพิ่มจาก 24 เป็น 32
            self.timesteps_multiplier = 10   # เพิ่มจาก 8 เป็น 10
            self.neural_network_size = "ultra"  # [8192, 8192, 4096, 2048, 1024]
            self.gpu_memory_fraction = 0.95  # ลดจาก 0.98 เป็น 0.95
            
        # Mid-range GPUs (8-16GB)
        elif self.gpu_memory_gb >= 8:
            self.max_concurrent_models = min(3, max(1, int(self.gpu_memory_gb / 3)))
            self.memory_per_model = 0.25  # 25% per model
            self.batch_size_multiplier = 4
            self.timesteps_multiplier = 2
            self.neural_network_size = "medium"  # [2048, 2048, 1024, 512]
            self.gpu_memory_fraction = 0.85
            
        # Low-end GPUs (4-8GB)
        elif self.gpu_memory_gb >= 4:
            self.max_concurrent_models = min(2, max(1, int(self.gpu_memory_gb / 3)))
            self.memory_per_model = 0.35  # 35% per model
            self.batch_size_multiplier = 2
            self.timesteps_multiplier = 1.5
            self.neural_network_size = "small"  # [1024, 1024, 512, 256]
            self.gpu_memory_fraction = 0.80
            
        # CPU-only or very low GPU memory
        else:
            self.max_concurrent_models = 1
            self.memory_per_model = 1.0
            self.batch_size_multiplier = 1
            self.timesteps_multiplier = 1
            self.neural_network_size = "minimal"  # [512, 512, 256]
            self.gpu_memory_fraction = 0.50 if self.gpu_available else 0
        
        # CPU Thread Optimization
        self.cpu_threads_training = min(16, max(4, self.cpu_cores))
        self.cpu_threads_interop = min(8, max(2, self.cpu_cores // 2))
        
        # Async Batch Configuration - OPTIMIZED FOR 4-MODEL CONCURRENT TRAINING
        self.default_batch_size = min(self.max_concurrent_models, 4)  # เพิ่มจาก 2 เป็น 4
        self.max_batch_size = min(4, self.max_concurrent_models)      # เพิ่มจาก 3 เป็น 4
        
        # Training Timeouts (seconds)
        self.model_timeout = 900   # เพิ่มจาก 300 เป็น 900 (15 minutes)
        self.batch_timeout = 1800  # เพิ่มจาก 600 เป็น 1800 (30 minutes)
        
        # Memory Management
        self.memory_cleanup_interval = 5  # Clean memory every 5 models
        self.force_gc_collection = True
        
    def get_neural_network_architecture(self):
        """Get neural network architecture based on hardware"""
        architectures = {
            "ultra": [32768, 16384, 8192, 4096, 2048, 1024],  # เพิ่มจาก [12288, 8192, ...] เป็น [32768, 16384, ...]
            "large": [16384, 8192, 4096, 2048, 1024],         # เพิ่มขนาด
            "medium": [8192, 4096, 2048, 1024],               # เพิ่มขนาด
            "small": [4096, 2048, 1024, 512],                 # เพิ่มขนาด
            "minimal": [2048, 1024, 512]   
        }
        return architectures.get(self.neural_network_size, architectures["medium"])
    
    def get_optimal_batch_size(self, base_batch_size=64):
        """Get optimal batch size for current hardware"""
        return min(base_batch_size * self.batch_size_multiplier, 32768)
    
    def get_optimal_timesteps(self, base_timesteps=100000):
        """Get optimal timesteps for current hardware"""
        return min(int(base_timesteps * self.timesteps_multiplier), 5000000)
    
    def apply_gpu_optimizations(self):
        """Apply GPU optimizations based on hardware"""
        if not self.gpu_available:
            return False
        
        try:
            # Memory fraction
            torch.cuda.set_per_process_memory_fraction(self.gpu_memory_fraction)
            
            # Environment variables
            os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = f'max_split_size_mb:{int(self.gpu_memory_gb * 128)},expandable_segments:True'  # เพิ่มจาก 32 เป็น 128
            os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            
            # PyTorch optimizations - MAXIMUM PERFORMANCE
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.enabled = True
            torch.backends.cuda.enable_math_sdp(True)  # เพิ่ม
            torch.backends.cuda.enable_flash_sdp(True)  # เพิ่ม
            
            # RTX 5060 TI specific optimizations
            if "RTX 5060" in self.gpu_name or "RTX 50" in self.gpu_name:
                try:
                    # Memory optimizations
                    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
                    torch.backends.cuda.cufft_plan_cache.max_size = 16384  # เพิ่มจาก 8192
                    torch.backends.cuda.preferred_linalg_library = "cusolver"
                    torch.backends.cuda.flash_attn_error_on_noncuda = False
                    
                    # Advanced compute optimizations
                    torch.set_float32_matmul_precision('high')  # เพิ่ม
                    torch.backends.cudnn.benchmark = True
                    torch.backends.cudnn.enabled = True
                    
                    # Memory pool optimizations
                    torch.cuda.empty_cache()
                    torch.cuda.memory.set_per_process_memory_fraction(self.gpu_memory_fraction)
                except:
                    pass  # Not all features available on all systems
            
            # CPU optimizations (threading configured globally at startup)
            try:
                torch.set_num_threads(self.cpu_threads_training)
                torch.set_num_interop_threads(self.cpu_threads_interop)
            except Exception:
                pass
            
            return True
            
        except Exception as e:
            print(f"⚠️ GPU optimization failed: {e}")
            return False
    def warmup_gpu(self):
        """Warmup GPU for maximum performance"""
        if not self.gpu_available:
            return
        
        try:
            print("🔥 Warming up RTX 5060 TI for maximum performance...")
            
            # Create large tensors to utilize GPU fully
            device = torch.device('cuda:0')
            
            # Warmup with progressively larger tensors
            for size in [1024, 2048, 4096, 8192]:
                x = torch.randn(size, size, device=device)
                y = torch.randn(size, size, device=device)
                
                # Matrix multiplications to warmup cores
                for _ in range(10):
                    z = torch.matmul(x, y)
                    z = torch.relu(z)
                
                del x, y, z
                torch.cuda.synchronize()
            
            # Warmup neural network layers
            net_arch = self.get_neural_network_architecture()
            dummy_input = torch.randn(256, net_arch[0], device=device)  # Large batch
            
            layers = []
            for i in range(len(net_arch)-1):
                layers.append(torch.nn.Linear(net_arch[i], net_arch[i+1]).to(device))
                layers.append(torch.nn.ReLU())
            
            # Forward pass warmup
            x = dummy_input
            for layer in layers:
                x = layer(x)
            
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            
            print("✅ RTX 5060 TI warmed up for maximum performance!")
            
        except Exception as e:
            print(f"⚠️ GPU warmup failed: {e}")
    
    def print_config(self):
        """Print current configuration"""
        print("🔧 ASYNC TRAINING CONFIGURATION")
        print("="*50)
        print(f"💻 CPU Cores: {self.cpu_cores}")
        print(f"🚀 GPU: {self.gpu_name}")
        print(f"💾 GPU Memory: {self.gpu_memory_gb:.1f} GB")
        print(f"🧠 System Memory: {self.system_memory_gb:.1f} GB")
        print()
        print(f"⚡ Async Settings:")
        print(f"   Max Concurrent Models: {self.max_concurrent_models}")
        print(f"   Memory per Model: {self.memory_per_model*100:.0f}%")
        print(f"   Default Batch Size: {self.default_batch_size}")
        print(f"   Max Batch Size: {self.max_batch_size}")
        print()
        print(f"🎯 Optimization Settings:")
        print(f"   Neural Network: {self.neural_network_size} {self.get_neural_network_architecture()}")
        print(f"   Batch Size Multiplier: {self.batch_size_multiplier}x")
        print(f"   Timesteps Multiplier: {self.timesteps_multiplier}x")
        print(f"   GPU Memory Fraction: {self.gpu_memory_fraction*100:.0f}%")
        print(f"   CPU Threads: {self.cpu_threads_training}")
                # Warmup GPU after printing config
        if self.gpu_available and "RTX 5060" in self.gpu_name:
            self.warmup_gpu()

# Global configuration instance
CONFIG = AsyncTrainingConfig()

def get_config():
    """Get global configuration"""
    return CONFIG

def print_system_info():
    """Print detailed system information"""
    config = get_config()
    config.print_config()
    
    if config.gpu_available:
        print(f"\n🔥 GPU Details:")
        print(f"   Device Count: {config.gpu_count}")
        print(f"   Compute Capability: {config.gpu_compute_capability}")
        print(f"   Memory Total: {config.gpu_memory_gb:.1f} GB")
        
        if torch.cuda.is_available():
            print(f"   Memory Free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**3:.1f} GB")
            print(f"   CUDA Version: {torch.version.cuda}")

if __name__ == "__main__":
    print_system_info()
