#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU Bottleneck Debug Script
ตรวจสอบปัญหา GPU utilization ต่ำ (17% instead of 80-95%)
"""

import asyncio
import psutil
import torch
import sys
import os

def check_system_resources():
    """ตรวจสอบทรัพยากรระบบ"""
    print("=== SYSTEM RESOURCES ===")
    print(f"CPU count: {psutil.cpu_count()}")
    print(f"Memory total: {psutil.virtual_memory().total // (1024**3)} GB")
    print(f"Memory available: {psutil.virtual_memory().available // (1024**3)} GB")
    
    # ตรวจสอบ Python processes ที่ใช้ memory สูง
    print("\n=== HIGH MEMORY PYTHON PROCESSES ===")
    for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
        try:
            if proc.info['name'] == 'python.exe':
                memory_mb = proc.info['memory_info'].rss // (1024*1024)
                print(f"PID {proc.info['pid']}: {memory_mb} MB")
        except:
            pass

def check_cuda_status():
    """ตรวจสอบสถานะ CUDA"""
    print("\n=== CUDA STATUS ===")
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA devices: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        
        # Memory usage
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        print(f"GPU Memory allocated: {allocated:.2f} GB")
        print(f"GPU Memory reserved: {reserved:.2f} GB")
        
        # Total GPU memory
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Total GPU Memory: {total_memory:.2f} GB")
        print(f"Memory usage percentage: {(reserved/total_memory)*100:.1f}%")
    else:
        print("❌ CUDA not available!")

def check_async_config():
    """ตรวจสอบการตั้งค่า async"""
    print("\n=== ASYNC TRAINING CONFIG ===")
    try:
        sys.path.append(os.getcwd())
        from async_config import AsyncTrainingConfig
        
        config = AsyncTrainingConfig()
        print(f"Max concurrent models: {config.max_concurrent_models}")
        print(f"Default batch size: {config.default_batch_size}")
        print(f"Max batch size: {config.max_batch_size}")
        print(f"Memory per model: {config.memory_per_model}")
        
        # RTX 5060 Ti specific
        if hasattr(config, 'rtx_5060_ti'):
            rtx_config = config.rtx_5060_ti
            print(f"RTX 5060 Ti memory per model: {rtx_config['memory_per_model']}")
            print(f"RTX 5060 Ti neural size: {rtx_config['neural_network_size']}")
        
        return config
    except Exception as e:
        print(f"❌ Error loading async config: {e}")
        return None

def diagnose_bottleneck(config):
    """วิเคราะห์ bottleneck"""
    print("\n=== BOTTLENECK DIAGNOSIS ===")
    
    # คำนวณ theoretical GPU utilization
    if config:
        concurrent_models = config.max_concurrent_models
        memory_per_model = config.memory_per_model
        
        print(f"Theoretical setup:")
        print(f"  - {concurrent_models} models running concurrently")
        print(f"  - {memory_per_model*100:.0f}% memory per model")
        print(f"  - Expected total GPU memory: {concurrent_models * memory_per_model * 100:.0f}%")
        
        if concurrent_models * memory_per_model > 1.0:
            print("⚠️  WARNING: Memory allocation exceeds 100%!")
            print("   This might cause models to wait for GPU memory")
        
        # ปัญหาที่เป็นไปได้
        print(f"\n🔍 POTENTIAL ISSUES:")
        print(f"1. Only 1 Python process using GPU (should be {concurrent_models})")
        print(f"2. GPU utilization 17% (should be 80-95%)")
        print(f"3. Memory usage 2GB (should be ~{16*concurrent_models*memory_per_model:.1f}GB)")
        
        # แนะนำการแก้ไข
        print(f"\n💡 RECOMMENDATIONS:")
        print(f"1. ตรวจสอบว่า async training กำลังทำงานจริงหรือไม่")
        print(f"2. ลด memory_per_model หากมี memory conflict")
        print(f"3. เพิ่ม batch size เพื่อใช้ GPU ให้เต็มที่")
        print(f"4. ตรวจสอบ queue และ worker processes")

def main():
    """Main diagnostic function"""
    print("🔍 GPU BOTTLENECK DIAGNOSTIC TOOL")
    print("=" * 50)
    
    check_system_resources()
    check_cuda_status() 
    config = check_async_config()
    diagnose_bottleneck(config)
    
    print("\n" + "=" * 50)
    print("✅ Diagnostic complete!")

if __name__ == "__main__":
    main()
