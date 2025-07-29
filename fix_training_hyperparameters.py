#!/usr/bin/env python3
"""
🔧 Fix Training Hyperparameters
แก้ไขปัญหา hyperparameters ที่ทำให้โมเดล collapse
"""

import json
import os
from datetime import datetime

def get_safe_hyperparameters():
    """
    ส่งคืน hyperparameters ที่ปลอดภัยและ stable
    แก้ไขปัญหาจาก train_all_models.py
    """
    
    print("🔧 GENERATING SAFE HYPERPARAMETERS")
    print("=" * 50)
    print("🚨 ปัญหาที่แก้ไข:")
    print("   1. ent_coef ลดจาก 1.0-2.0 → 0.01-0.05")
    print("   2. clip_range ลดจาก 0.25 → 0.1-0.2") 
    print("   3. learning_rate ลดจาก 0.001 → 0.0001-0.0003")
    print("   4. reward function ทำให้สมดุลขึ้น")
    print("   5. เอา forced trading ออก")
    print("")
    
    # 🔧 SAFE PPO Configuration
    safe_ppo_config = {
        'algorithm': 'PPO',
        'learning_rate': 0.0003,           # ⬇️ ลดจาก 0.001
        'gamma': 0.99,                     # ✅ Standard value
        'n_steps': 2048,                   # ⬇️ ลดจาก 8192
        'batch_size': 256,                 # ⬇️ ลดจาก 2048+
        'n_epochs': 4,                     # ⬇️ ลดจาก 15
        'clip_range': 0.2,                 # ⬇️ ลดจาก 0.25
        'ent_coef': 0.01,                  # ⬇️ ลดจาก 1.0-2.0 ไป 0.01
        'vf_coef': 0.5,                    # ✅ Standard
        'max_grad_norm': 0.5,              # ✅ Standard
        'lookback_window': 50,             # ✅ Reasonable
        'transaction_cost': 0.0001,        # ✅ Realistic
        'timesteps': 1000000               # ⬇️ ลดจาก 5M
    }
    
    # 🔧 SAFE SAC Configuration  
    safe_sac_config = {
        'algorithm': 'SAC',
        'learning_rate': 0.0003,           # ⬇️ ลดจาก 0.001
        'gamma': 0.99,                     # ✅ Standard
        'batch_size': 256,                 # ⬇️ ลดจาก 512+
        'buffer_size': 100000,             # ⬇️ ลดจาก 1M
        'learning_starts': 1000,           # ✅ Standard
        'tau': 0.005,                      # ⬇️ ลดจาก 0.02
        'ent_coef': 0.2,                   # ⬇️ ลดจาก 1.2 → 0.2 (SAC standard)
        'target_update_interval': 1,       # ✅ Standard
        'gradient_steps': 1,               # ✅ Standard
        'lookback_window': 50,             # ✅ Reasonable
        'transaction_cost': 0.0001,        # ✅ Realistic
        'timesteps': 1000000               # ⬇️ ลดจาก 5M
    }
    
    return safe_ppo_config, safe_sac_config

def get_safe_reward_guidelines():
    """แนวทางการออกแบบ reward function ที่ปลอดภัย"""
    
    guidelines = {
        "reward_principles": {
            "1_balanced_range": "Reward ควรอยู่ในช่วง -10 ถึง +10 ไม่ควรเกิน ±50",
            "2_gradual_penalties": "Penalty ค่อยเป็นค่อยไป ไม่ควร jump จาก 0 → -50",
            "3_positive_reinforcement": "ใช้ positive reward มากกว่า negative penalty",
            "4_realistic_goals": "ตั้งเป้าหมายที่สมเหตุสมผล ไม่บังคับเทรดทุก step",
            "5_stability_first": "เน้น stability ก่อน aggressive trading"
        },
        
        "reward_ranges": {
            "profitable_trade": "+2 ถึง +8 (ไม่ควรเกิน +10)",
            "losing_trade": "-1 ถึง -5 (ไม่ควรเกิน -10)", 
            "hold_action": "-0.1 ถึง -1 (เล็กน้อย)",
            "invalid_action": "-2 ถึง -5",
            "episode_bonus": "+5 ถึง +15 (สำหรับ final performance)"
        },
        
        "avoid_patterns": [
            "ห้าม reward/penalty เกิน ±20 ใน single step",
            "ห้าม forced random trading",
            "ห้าม massive penalty สำหรับ hold",
            "ห้าม extreme win rate requirements (>80%)",
            "ห้าม unrealistic profit targets"
        ]
    }
    
    return guidelines

def create_safe_training_config():
    """สร้าง training config ที่ปลอดภัย"""
    
    safe_ppo, safe_sac = get_safe_hyperparameters()
    guidelines = get_safe_reward_guidelines()
    
    config = {
        "meta": {
            "created": datetime.now().isoformat(),
            "purpose": "Safe hyperparameters to prevent model collapse",
            "fixes_applied": [
                "Reduced entropy coefficient from 1.0-2.0 to 0.01-0.2",
                "Reduced clip_range from 0.25 to 0.2",
                "Reduced learning_rate from 0.001 to 0.0003", 
                "Reduced batch_size for stability",
                "Removed forced trading logic",
                "Balanced reward function"
            ]
        },
        
        "recommended_configs": {
            "ppo_safe": safe_ppo,
            "sac_safe": safe_sac
        },
        
        "reward_guidelines": guidelines,
        
        "training_recommendations": {
            "start_with": "PPO with safe config",
            "training_duration": "1M timesteps first, then extend if stable",
            "monitoring": "Watch for bias_counter and model variance",
            "early_stopping": "Stop if win_rate < 40% after 500k timesteps",
            "validation": "Test on paper trading before live deployment"
        }
    }
    
    return config

def save_safe_config():
    """บันทึก safe configuration"""
    
    config = create_safe_training_config()
    
    filename = f"safe_training_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = os.path.join("training_configs", filename)
    
    # สร้างโฟลเดอร์ถ้าไม่มี
    os.makedirs("training_configs", exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"✅ บันทึก safe config ที่: {filepath}")
    
    return filepath, config

def compare_problematic_vs_safe():
    """เปรียบเทียบ hyperparameters ที่มีปัญหากับแบบปลอดภัย"""
    
    print("\n🔍 HYPERPARAMETER COMPARISON")
    print("=" * 60)
    
    comparison = [
        ("Parameter", "❌ Problematic", "✅ Safe", "📝 Impact"),
        ("─" * 15, "─" * 15, "─" * 10, "─" * 25),
        ("ent_coef", "1.0-2.0", "0.01", "Reduces random actions"),
        ("clip_range", "0.25", "0.2", "More stable updates"),
        ("learning_rate", "0.001", "0.0003", "Prevents overshooting"),
        ("batch_size", "2048+", "256", "Better gradient estimates"),
        ("reward_range", "±50", "±10", "Balanced learning signals"),
        ("forced_trading", "Yes", "No", "Natural behavior learning"),
        ("timesteps", "5M", "1M", "Faster experimentation")
    ]
    
    for row in comparison:
        print(f"{row[0]:<15} | {row[1]:<15} | {row[2]:<10} | {row[3]}")
    
    print("\n🎯 KEY INSIGHTS:")
    print("   • entropy coefficient (ent_coef) เป็นสาเหตุหลักของ random behavior")
    print("   • ค่า reward ที่สุดโต่งทำให้ model เรียนรู้ extreme actions")
    print("   • learning rate สูงเกินไปทำให้ไม่ stable")
    print("   • forced trading ทำลาย natural learning process")

if __name__ == "__main__":
    print("🔧 TRAINING HYPERPARAMETER ANALYSIS & FIXES")
    print("=" * 55)
    
    # แสดงการเปรียบเทียบ
    compare_problematic_vs_safe()
    
    # สร้างและบันทึก safe config
    filepath, config = save_safe_config()
    
    print(f"\n📋 NEXT STEPS:")
    print(f"   1. 🛑 Stop current training immediately")
    print(f"   2. 📝 Use safe config: {filepath}")
    print(f"   3. 🔄 Retrain with balanced hyperparameters") 
    print(f"   4. 📊 Monitor bias_counter and model variance")
    print(f"   5. 🧪 Validate thoroughly before live trading")
    
    print(f"\n🎯 ROOT CAUSE SUMMARY:")
    print(f"   • Model collapse เกิดจาก extreme hyperparameters")
    print(f"   • ent_coef=1.0-2.0 ทำให้ AI เลือก action แบบสุ่ม")
    print(f"   • reward function สุดโต่งทำให้เรียนรู้ extreme behavior")
    print(f"   • forced trading ทำลาย natural learning")
