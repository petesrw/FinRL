#!/usr/bin/env python3
"""
Training Loop Analysis Tool
วิเคราะห์จำนวน loop และ batch ในการเทรน
"""

import json
import os
from datetime import datetime

def analyze_training_loops():
    """วิเคราะห์จำนวน loop/batch ที่เทรนไปแล้ว"""
    print("📊 ASYNC TRAINING LOOP ANALYSIS")
    print("=" * 60)
    
    # อ่าน async training log
    log_file = 'training_logs/async_logs/xauusd_async_training.json'
    
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            logs = json.load(f)
        
        total_batches = len(logs)
        total_models = 0
        total_successful = 0
        best_score = 0
        best_tier = "none"
        
        print(f"📁 Log file: {log_file}")
        print(f"📅 Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        for batch in logs:
            batch_num = batch.get('batch_number', 0)
            models_in_batch = batch.get('total_models', 0)
            successful_in_batch = batch.get('successful_models', 0)
            batch_timestamp = batch.get('timestamp', 'Unknown')
            
            total_models += models_in_batch
            total_successful += successful_in_batch
            
            print(f"🔄 Batch {batch_num} ({batch_timestamp[:19]}):")
            print(f"   📈 Models trained: {models_in_batch}")
            print(f"   ✅ Successful: {successful_in_batch}")
            print(f"   ❌ Failed: {models_in_batch - successful_in_batch}")
            
            # หา best score ใน batch นี้
            batch_best_score = 0
            batch_best_tier = "none"
            
            results = batch.get('results', [])
            for result in results:
                if result.get('success', False):
                    score = result.get('score', 0)
                    tier = result.get('tier', 'none')
                    
                    if score > batch_best_score:
                        batch_best_score = score
                        batch_best_tier = tier
                    
                    if score > best_score:
                        best_score = score
                        best_tier = tier
            
            if batch_best_score > 0:
                print(f"   🏆 Best in batch: {batch_best_score:.1f} ({batch_best_tier.upper()})")
            
            print()
        
        # สรุปผลรวม
        print("📊 TOTAL SUMMARY:")
        print(f"   🔢 Total Batches: {total_batches}")
        print(f"   🎯 Total Models Trained: {total_models}")
        print(f"   ✅ Total Successful: {total_successful}")
        print(f"   ❌ Total Failed: {total_models - total_successful}")
        if total_models > 0:
            print(f"   📈 Success Rate: {total_successful/total_models*100:.1f}%")
        
        print(f"\n🏆 BEST OVERALL PERFORMANCE:")
        print(f"   📊 Best Score: {best_score:.1f}")
        print(f"   🥇 Best Tier: {best_tier.upper()}")
        
        # คำนวณ loop rate
        if total_batches > 0:
            avg_models_per_batch = total_models / total_batches
            print(f"\n⚡ TRAINING EFFICIENCY:")
            print(f"   🔄 Average models per batch: {avg_models_per_batch:.1f}")
            print(f"   🚀 Total training loops: {total_models}")
            
            # ประมาณเวลาที่ใช้
            if total_batches > 1:
                first_batch_time = datetime.fromisoformat(logs[0]['timestamp'])
                last_batch_time = datetime.fromisoformat(logs[-1]['timestamp'])
                duration = last_batch_time - first_batch_time
                
                hours = duration.total_seconds() / 3600
                if hours > 0:
                    models_per_hour = total_models / hours
                    print(f"   ⏱️ Training duration: {hours:.1f} hours")
                    print(f"   🔥 Models per hour: {models_per_hour:.1f}")
        
    else:
        print("❌ No async training logs found")
        print("   File expected at: training_logs/async_logs/xauusd_async_training.json")
        
        # ลองหาไฟล์อื่น
        if os.path.exists('training_logs'):
            print("\n📁 Available log directories:")
            for item in os.listdir('training_logs'):
                if os.path.isdir(os.path.join('training_logs', item)):
                    print(f"   📂 {item}/")
        else:
            print("   📁 training_logs/ directory not found")

if __name__ == "__main__":
    analyze_training_loops()
