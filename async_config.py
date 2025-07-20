#!/usr/bin/env python3
"""
🚀 Async Training Configuration
Advanced settings for async multi-model training
"""

import torch
import multiprocessing as mp
import os

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
        
        # RTX 5060 TI Specific Optimization
        if "RTX 5060" in self.gpu_name or "RTX 50" in self.gpu_name:
            self.max_concurrent_models = min(6, max(2, int(self.gpu_memory_gb / 2.5)))
            self.memory_per_model = 0.20  # 15% per model for RTX 5060 TI
            self.batch_size_multiplier = 8
            self.timesteps_multiplier = 4
            self.neural_network_size = "ultra"  # [8192, 8192, 4096, 2048, 1024]
            self.gpu_memory_fraction = 0.95  # Use 95% of 16GB
            
        # High-end GPUs (16GB+)
        elif self.gpu_memory_gb >= 16:
            self.max_concurrent_models = min(4, max(2, int(self.gpu_memory_gb / 4)))
            self.memory_per_model = 0.20  # 20% per model
            self.batch_size_multiplier = 6
            self.timesteps_multiplier = 3
            self.neural_network_size = "large"  # [4096, 4096, 2048, 1024, 512]
            self.gpu_memory_fraction = 0.90
            
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
        
        # Async Batch Configuration
        self.default_batch_size = min(self.max_concurrent_models, 4)
        self.max_batch_size = min(8, self.max_concurrent_models)
        
        # Training Timeouts (seconds)
        self.model_timeout = 300  # 5 minutes per model
        self.batch_timeout = self.model_timeout * 2  # 10 minutes per batch
        
        # Memory Management
        self.memory_cleanup_interval = 5  # Clean memory every 5 models
        self.force_gc_collection = True
        
    def get_neural_network_architecture(self):
        """Get neural network architecture based on hardware"""
        architectures = {
            "ultra": [8192, 8192, 4096, 2048, 1024],
            "large": [4096, 4096, 2048, 1024, 512],
            "medium": [2048, 2048, 1024, 512],
            "small": [1024, 1024, 512, 256],
            "minimal": [512, 512, 256]
        }
        return architectures.get(self.neural_network_size, architectures["medium"])
    
    def get_optimal_batch_size(self, base_batch_size=64):
        """Get optimal batch size for current hardware"""
        return min(base_batch_size * self.batch_size_multiplier, 8192)
    
    def get_optimal_timesteps(self, base_timesteps=100000):
        """Get optimal timesteps for current hardware"""
        return min(int(base_timesteps * self.timesteps_multiplier), 500000)
    
    def apply_gpu_optimizations(self):
        """Apply GPU optimizations based on hardware"""
        if not self.gpu_available:
            return False
        
        try:
            # Memory fraction
            torch.cuda.set_per_process_memory_fraction(self.gpu_memory_fraction)
            
            # Environment variables
            os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = f'max_split_size_mb:{int(self.gpu_memory_gb * 32)}'
            
            # PyTorch optimizations
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # RTX 5060 TI specific optimizations
            if "RTX 5060" in self.gpu_name or "RTX 50" in self.gpu_name:
                try:
                    torch.backends.cuda.enable_flash_sdp(True)
                    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
                    torch.backends.cuda.cufft_plan_cache.max_size = 8192
                    torch.backends.cuda.preferred_linalg_library = "cusolver"
                except:
                    pass  # Not all features available on all systems
            
            # CPU optimizations
            torch.set_num_threads(self.cpu_threads_training)
            torch.set_num_interop_threads(self.cpu_threads_interop)
            
            return True
            
        except Exception as e:
            print(f"⚠️ GPU optimization failed: {e}")
            return False
    
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
