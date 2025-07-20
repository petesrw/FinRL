import json
import os

print("🔍 ตรวจสอบแหล่งข้อมูล Failed Configurations")
print("="*50)

# 1. ตรวจสอบ training history
history_file = 'training_logs/history/xauusd_training_history.json'
if os.path.exists(history_file):
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    failed_from_history = 0
    print(f"📚 Training History: {len(history)} entries")
    
    for h in history[:5]:  # แสดง 5 รายการแรก
        score = h.get('score', 0)
        tier = h.get('tier', 'unknown')
        has_error = 'error' in h
        is_failed = score < 40 or tier == 'failed' or has_error
        
        if is_failed:
            failed_from_history += 1
        
        print(f"  - Attempt {h['attempt']}: Score {score}, Tier {tier}, Failed: {is_failed}")
    
    # นับทั้งหมด
    total_failed_history = sum(1 for h in history if h.get('score', 0) < 40 or h.get('tier') == 'failed' or 'error' in h)
    print(f"  🚫 Total failed from history: {total_failed_history}")
else:
    print("❌ History file not found")
    total_failed_history = 0

# 2. ตรวจสอบ failed configs file
failed_file = 'training_logs/failed_configs/xauusd_failed_configs.json'
if os.path.exists(failed_file):
    with open(failed_file, 'r') as f:
        failed_configs = json.load(f)
    print(f"📁 Failed Configs File: {len(failed_configs)} entries")
    total_failed_file = len(failed_configs)
else:
    print("📁 Failed Configs File: Not found")
    total_failed_file = 0

print(f"")
print(f"📊 สรุป:")
print(f"   History file: {total_failed_history} failed configs")
print(f"   Failed file: {total_failed_file} failed configs")
print(f"   รวมทั้งหมด: {total_failed_history + total_failed_file} configurations")
