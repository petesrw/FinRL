#!/usr/bin/env python3
"""
Real-time GPU Utilization Monitor
Monitor GPU usage during training to validate bottleneck fixes
"""

import subprocess
import time
import json
import os
from datetime import datetime

def get_gpu_stats():
    """Get current GPU statistics"""
    try:
        result = subprocess.run([
            'nvidia-smi', 
            '--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            values = result.stdout.strip().split(', ')
            return {
                'gpu_util': int(values[0]),
                'memory_used': int(values[1]),
                'memory_total': int(values[2]),
                'power_draw': float(values[3]),
                'temperature': int(values[4]),
                'memory_percent': round((int(values[1])/int(values[2]))*100, 1),
                'timestamp': datetime.now().strftime('%H:%M:%S')
            }
    except Exception as e:
        print(f"Error getting GPU stats: {e}")
        return None

def get_python_processes():
    """Get Python process count and memory usage"""
    try:
        result = subprocess.run([
            'tasklist', '/FI', 'IMAGENAME eq python.exe', '/FO', 'CSV'
        ], capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            processes = []
            for line in lines:
                if line.strip():
                    parts = line.replace('"', '').split(',')
                    if len(parts) >= 5:
                        memory_kb = parts[4].replace(' K', '').replace(',', '')
                        if memory_kb.isdigit():
                            processes.append({
                                'pid': parts[1],
                                'memory_mb': int(memory_kb) // 1024
                            })
            return processes
    except:
        return []

def monitor_training(duration=60, interval=3):
    """Monitor training for specified duration"""
    print("🔍 GPU UTILIZATION MONITOR - Bottleneck Analysis")
    print("=" * 60)
    print("Time     GPU%  Memory%   Power  Temp  Processes  Max Memory")
    print("-" * 60)
    
    max_gpu_util = 0
    max_memory_util = 0
    max_power = 0
    
    for i in range(duration // interval):
        stats = get_gpu_stats()
        processes = get_python_processes()
        
        if stats:
            # Track maximums
            max_gpu_util = max(max_gpu_util, stats['gpu_util'])
            max_memory_util = max(max_memory_util, stats['memory_percent'])
            max_power = max(max_power, stats['power_draw'])
            
            # Display current stats
            process_count = len(processes)
            max_proc_memory = max([p['memory_mb'] for p in processes], default=0)
            
            print(f"{stats['timestamp']}  {stats['gpu_util']:3d}%   {stats['memory_percent']:5.1f}%   {stats['power_draw']:5.1f}W  {stats['temperature']:2d}°C     {process_count:3d}      {max_proc_memory:,}MB")
            
            # Check for significant improvements
            if stats['gpu_util'] > 50:
                print(f"  🚀 HIGH GPU UTILIZATION: {stats['gpu_util']}%")
            
            if len(processes) > 1:
                print(f"  📊 MULTIPLE PROCESSES: {process_count} Python processes")
        
        time.sleep(interval)
    
    print("-" * 60)
    print("📈 MONITORING SUMMARY:")
    print(f"   Max GPU Utilization: {max_gpu_util}%")
    print(f"   Max Memory Usage: {max_memory_util:.1f}%") 
    print(f"   Max Power Draw: {max_power:.1f}W")
    
    # Analysis
    if max_gpu_util < 30:
        print(f"⚠️  LOW GPU UTILIZATION - Possible issues:")
        print(f"   - CPU-bound operations (data loading/preprocessing)")
        print(f"   - Small batch sizes")
        print(f"   - Sequential rather than concurrent training")
        print(f"   - Memory allocation conflicts")
    elif max_gpu_util > 70:
        print(f"✅ EXCELLENT GPU UTILIZATION - Bottleneck likely resolved!")
    else:
        print(f"📊 MODERATE GPU UTILIZATION - Some improvement")

if __name__ == "__main__":
    print("Starting 60-second GPU monitoring...")
    print("Watch for spikes during actual model training phases")
    print()
    monitor_training(duration=60, interval=3)
