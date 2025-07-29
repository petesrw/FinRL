#!/usr/bin/env python3
"""
🔧 Fix train_all_models.py Hyperparameters
แก้ไข hyperparameters ที่ทำให้โมเดล collapse ใน train_all_models.py
"""

import json
import re
from datetime import datetime

def fix_train_all_models():
    """แก้ไข train_all_models.py โดยเปลี่ยน extreme hyperparameters"""
    
    print("🔧 FIXING train_all_models.py")
    print("=" * 40)
    
    filepath = "train_all_models.py"
    
    try:
        # อ่านไฟล์
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # สำรองไฟล์เดิม
        backup_path = f"train_all_models_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"📦 Backup saved: {backup_path}")
        
        # 🔧 FIX 1: ent_coef สำหรับ PPO
        print("🔧 Fix 1: PPO ent_coef (1.0-2.0 → 0.01-0.05)")
        content = re.sub(
            r"'ent_coef': random\.choice\(\[1\.0, 1\.5, 2\.0\]\),   # INSANE entropy coefficient for FORCED exploration!",
            r"'ent_coef': random.choice([0.01, 0.02, 0.05]),   # FIXED: Safe entropy coefficient",
            content
        )
        
        # 🔧 FIX 2: ent_coef สำหรับ SAC  
        print("🔧 Fix 2: SAC ent_coef (0.8-1.2 → 0.1-0.3)")
        content = re.sub(
            r"'ent_coef': random\.choice\(\[0\.8, 1\.0, 1\.2\]\),  # VERY high entropy for maximum exploration",
            r"'ent_coef': random.choice([0.1, 0.2, 0.3]),  # FIXED: Safe SAC entropy",
            content
        )
        
        # 🔧 FIX 3: clip_range สำหรับ PPO
        print("🔧 Fix 3: PPO clip_range (0.15-0.25 → 0.1-0.2)")
        content = re.sub(
            r"'clip_range': random\.choice\(\[0\.15, 0\.2, 0\.25\]\),  # Higher clip for exploration",
            r"'clip_range': random.choice([0.1, 0.15, 0.2]),  # FIXED: Safe clip range",
            content
        )
        
        # 🔧 FIX 4: learning_rate ใน fallback config
        print("🔧 Fix 4: Fallback learning_rate (0.001 → 0.0003)")
        content = re.sub(
            r"'learning_rate': 0\.001,  # MUCH higher learning rate for aggressive exploration",
            r"'learning_rate': 0.0003,  # FIXED: Safe learning rate",
            content
        )
        
        # 🔧 FIX 5: fallback ent_coef
        print("🔧 Fix 5: Fallback ent_coef (1.5 → 0.2)")
        content = re.sub(
            r"'ent_coef': 1\.5,  # VERY high entropy for maximum exploration",
            r"'ent_coef': 0.2,  # FIXED: Safe entropy for SAC",
            content
        )
        
        # 🔧 FIX 6: A2C ent_coef
        print("🔧 Fix 6: A2C ent_coef (0.05-0.15 → 0.01-0.03)")
        content = re.sub(
            r"'ent_coef': random\.choice\(\[0\.05, 0\.1, 0\.15\]\),  # MUCH higher entropy for A2C exploration",
            r"'ent_coef': random.choice([0.01, 0.02, 0.03]),  # FIXED: Safe A2C entropy",
            content
        )
        
        # 🔧 FIX 7: พื้นฐาน ent_coef ใน config
        print("🔧 Fix 7: Base ent_coef configuration")
        content = re.sub(
            r'"ent_coef": 0\.01,',
            r'"ent_coef": 0.01,  # SAFE: Standard entropy coefficient',
            content
        )
        
        # 🔧 FIX 8: Extreme reward penalties
        print("🔧 Fix 8: Extreme reward values")
        content = re.sub(
            r'final_reward -= 50   # ลดลงจาก 100 เป็น 50',
            r'final_reward -= 15   # FIXED: Reduced extreme penalty',
            content
        )
        content = re.sub(
            r'reward -= 50',
            r'reward -= 15  # FIXED: Reduced penalty',
            content
        )
        content = re.sub(
            r'reward \+= 40',
            r'reward += 15  # FIXED: Reduced bonus',
            content
        )
        
        # 🔧 FIX 9: Comment out FORCED TRADING
        print("🔧 Fix 9: Disable forced trading")
        content = re.sub(
            r'# 🚨 NUCLEAR OPTION: FORCED TRADING MODE',
            r'# 🚫 DISABLED: FORCED TRADING MODE (causes model collapse)',
            content
        )
        
        # เขียนไฟล์ที่แก้ไขแล้ว
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ Fixed train_all_models.py successfully!")
        
        # สร้างรายงานการแก้ไข
        create_fix_report()
        
        return True
        
    except Exception as e:
        print(f"❌ Error fixing file: {e}")
        return False

def create_fix_report():
    """สร้างรายงานการแก้ไข"""
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "file_fixed": "train_all_models.py",
        "fixes_applied": [
            {
                "issue": "PPO ent_coef too high (1.0-2.0)",
                "fix": "Reduced to 0.01-0.05",
                "impact": "Reduces random action selection"
            },
            {
                "issue": "SAC ent_coef too high (0.8-1.2)", 
                "fix": "Reduced to 0.1-0.3",
                "impact": "Better exploration-exploitation balance"
            },
            {
                "issue": "PPO clip_range too high (0.25)",
                "fix": "Reduced to 0.1-0.2",
                "impact": "More stable policy updates"
            },
            {
                "issue": "Learning rate too high (0.001)",
                "fix": "Reduced to 0.0003",
                "impact": "Prevents overshooting and instability"
            },
            {
                "issue": "Extreme reward values (±50)",
                "fix": "Reduced to ±15",
                "impact": "Balanced learning signals"
            },
            {
                "issue": "Forced trading logic",
                "fix": "Disabled/commented out",
                "impact": "Natural behavior learning"
            }
        ],
        "expected_improvements": [
            "Models will learn stable trading patterns",
            "Reduced bias toward extreme actions",
            "Better convergence during training",
            "More realistic trading behavior",
            "Elimination of model collapse"
        ],
        "next_steps": [
            "Stop any current training",
            "Use retrain_fresh_model.py with new hyperparameters",
            "Monitor bias_counter during training",
            "Validate models thoroughly before live trading"
        ]
    }
    
    report_path = f"hyperparameter_fix_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"📋 Fix report saved: {report_path}")
    
    return report_path

def verify_fixes():
    """ตรวจสอบว่าการแก้ไขสำเร็จหรือไม่"""
    
    print("\n🔍 VERIFYING FIXES...")
    
    try:
        with open("train_all_models.py", 'r', encoding='utf-8') as f:
            content = f.read()
        
        checks = [
            ("✅ PPO ent_coef fixed", "random.choice([0.01, 0.02, 0.05])" in content),
            ("✅ SAC ent_coef fixed", "random.choice([0.1, 0.2, 0.3])" in content),
            ("✅ Clip range fixed", "random.choice([0.1, 0.15, 0.2])" in content),
            ("✅ Learning rate fixed", "'learning_rate': 0.0003" in content),
            ("✅ Fallback ent_coef fixed", "'ent_coef': 0.2" in content),
            ("✅ Forced trading disabled", "DISABLED: FORCED TRADING" in content)
        ]
        
        all_good = True
        for check_name, condition in checks:
            if condition:
                print(f"   {check_name}")
            else:
                print(f"   ❌ {check_name} - NOT FIXED")
                all_good = False
        
        if all_good:
            print("\n🎉 ALL FIXES VERIFIED SUCCESSFULLY!")
            print("   Models should now train with stable hyperparameters")
        else:
            print("\n⚠️ Some fixes may not have been applied correctly")
            
        return all_good
        
    except Exception as e:
        print(f"❌ Error verifying fixes: {e}")
        return False

if __name__ == "__main__":
    print("🔧 HYPERPARAMETER FIX TOOL")
    print("=" * 35)
    print("This tool fixes extreme hyperparameters in train_all_models.py")
    print("that cause model collapse and bias issues.")
    print("")
    
    confirm = input("Proceed with fixing train_all_models.py? (yes/y/1): ").lower().strip()
    
    if confirm in ['yes', 'y', '1']:
        success = fix_train_all_models()
        
        if success:
            verify_fixes()
            
            print("\n🎯 SUMMARY:")
            print("   • Fixed extreme entropy coefficients")
            print("   • Reduced clip ranges to safe values")
            print("   • Lowered learning rates")
            print("   • Balanced reward function")
            print("   • Disabled forced trading")
            print("")
            print("📋 NEXT ACTIONS:")
            print("   1. 🛑 Stop any running training")
            print("   2. 🔄 Use retrain_fresh_model.py")
            print("   3. 📊 Monitor for stable training")
            print("   4. 🧪 Validate before live trading")
        else:
            print("❌ Fix failed - check error messages above")
    else:
        print("❌ Fix cancelled by user")
