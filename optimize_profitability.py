#!/usr/bin/env python3
"""
🚀 Optimize Profitability Hyperparameters
เพิ่มความสามารถในการทำกำไรโดยปรับสมดุลระหว่าง safety และ profitability
"""

import json
import os
from datetime import datetime

def create_optimized_profitability_config():
    """สร้าง config ที่เพิ่มความสามารถในการทำกำไร"""
    
    # 🎯 BALANCED AGGRESSIVE HYPERPARAMETERS
    # ไม่ extreme แต่เพิ่ม exploration และ risk-taking
    optimized_configs = [
        {
            "name": "Balanced Aggressive PPO",
            "hyperparameters": {
                "algorithm": "PPO",
                "learning_rate": 0.0005,  # เพิ่มจาก 0.0003
                "gamma": 0.95,  # ลดลงเล็กน้อยเพื่อ short-term focus
                "lookback_window": 50,
                "transaction_cost": 0,
                "timesteps": 2500000,  # เพิ่ม timesteps
                "n_steps": 4096,  # เพิ่มจาก 2048
                "batch_size": 1024,
                "n_epochs": 12,  # ลดลงเล็กน้อย
                "clip_range": 0.18,  # เพิ่มจาก 0.15
                "ent_coef": 0.03,  # เพิ่มจาก 0.01 เพื่อ exploration
                "vf_coef": 0.5,
                "max_grad_norm": 0.5
            },
            "expected_improvement": "เพิ่ม exploration แต่ยังคง stability"
        },
        {
            "name": "High Exploration PPO",
            "hyperparameters": {
                "algorithm": "PPO",
                "learning_rate": 0.0007,  # เพิ่มขึ้น
                "gamma": 0.96,
                "lookback_window": 75,  # เพิ่ม context
                "transaction_cost": 0,
                "timesteps": 3000000,
                "n_steps": 6144,  # เพิ่มขึ้นมาก
                "batch_size": 1536,  # เพิ่มขึ้น
                "n_epochs": 10,
                "clip_range": 0.2,  # กลับไปใช้ 0.2
                "ent_coef": 0.05,  # เพิ่มการ exploration
                "vf_coef": 0.6,  # เพิ่มเล็กน้อย
                "max_grad_norm": 0.6
            },
            "expected_improvement": "เพิ่ม exploration และ learning capacity"
        },
        {
            "name": "SAC Aggressive",
            "hyperparameters": {
                "algorithm": "SAC",
                "learning_rate": 0.0008,
                "gamma": 0.97,
                "lookback_window": 50,
                "transaction_cost": 0,
                "timesteps": 2500000,
                "batch_size": 512,  # SAC ใช้ batch เล็กกว่า
                "buffer_size": 1000000,
                "learning_starts": 1000,
                "tau": 0.01,  # เพิ่มจาก 0.005
                "ent_coef": 0.3,  # SAC entropy coefficient
                "target_update_interval": 1
            },
            "expected_improvement": "SAC มี natural exploration, เหมาะสำหรับ continuous action"
        },
        {
            "name": "Moderate Risk PPO",
            "hyperparameters": {
                "algorithm": "PPO", 
                "learning_rate": 0.0004,
                "gamma": 0.97,  # กลับมาใช้ 0.97
                "lookback_window": 60,
                "transaction_cost": 0,
                "timesteps": 2000000,
                "n_steps": 3072,  # ระหว่าง 2048 และ 4096
                "batch_size": 1024,
                "n_epochs": 15,
                "clip_range": 0.17,  # ระหว่าง 0.15 และ 0.2
                "ent_coef": 0.02,  # ระหว่าง 0.01 และ 0.03
                "vf_coef": 0.5,
                "max_grad_norm": 0.5
            },
            "expected_improvement": "สมดุลระหว่าง safety และ profitability"
        }
    ]
    
    return optimized_configs

def update_training_file():
    """อัพเดทไฟล์ train_all_models.py ด้วย optimized parameters"""
    
    backup_file = f"train_all_models_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
    
    print("🔄 Creating backup of train_all_models.py...")
    
    # สร้าง backup
    if os.path.exists("train_all_models.py"):
        with open("train_all_models.py", 'r', encoding='utf-8') as f:
            content = f.read()
        with open(backup_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Backup created: {backup_file}")
    
    # สร้าง updated hyperparameter generation
    updated_hyperparams = '''
    def generate_optimized_profitability_hyperparameters(self):
        """🚀 Generate Optimized Profitability Hyperparameters"""
        
        # 🎯 BALANCED AGGRESSIVE CONFIGURATIONS
        configs = [
            # Balanced Aggressive PPO
            {
                'algorithm': 'PPO',
                'learning_rate': random.choice([0.0005, 0.0006, 0.0007]),
                'gamma': random.choice([0.95, 0.96, 0.97]),
                'lookback_window': random.choice([50, 60, 75]),
                'transaction_cost': 0,
                'timesteps': random.choice([2500000, 3000000]),
                'n_steps': random.choice([3072, 4096, 6144]),
                'batch_size': random.choice([1024, 1536]),
                'n_epochs': random.choice([10, 12, 15]),
                'clip_range': random.choice([0.17, 0.18, 0.2]),
                'ent_coef': random.choice([0.02, 0.03, 0.04, 0.05]),  # เพิ่ม exploration
                'vf_coef': random.choice([0.5, 0.6]),
                'max_grad_norm': random.choice([0.5, 0.6])
            },
            # SAC Aggressive
            {
                'algorithm': 'SAC',
                'learning_rate': random.choice([0.0006, 0.0008, 0.001]),
                'gamma': random.choice([0.96, 0.97, 0.98]),
                'lookback_window': random.choice([50, 75]),
                'transaction_cost': 0,
                'timesteps': random.choice([2500000, 3000000]),
                'batch_size': random.choice([512, 768]),
                'buffer_size': random.choice([800000, 1000000]),
                'learning_starts': random.choice([1000, 1500]),
                'tau': random.choice([0.008, 0.01, 0.012]),
                'ent_coef': random.choice([0.2, 0.3, 0.4]),  # SAC entropy
                'target_update_interval': 1
            }
        ]
        
        # เลือก config แบบสุ่ม
        config = random.choice(configs)
        
        print(f"   🚀 Generated OPTIMIZED PROFITABILITY config:")
        print(f"   💡 Algorithm: {config['algorithm']}")
        print(f"   📈 Learning Rate: {config['learning_rate']}")
        print(f"   🎯 Entropy Coef: {config['ent_coef']}")
        print(f"   🔄 Timesteps: {config['timesteps']:,}")
        
        return config
'''
    
    return updated_hyperparams

def create_quick_retrain_script():
    """สร้างสคริปต์สำหรับ retrain ด้วย optimized parameters"""
    
    script_content = '''#!/usr/bin/env python3
"""
🚀 Quick Retrain with Optimized Profitability Parameters
"""

import sys
import os
sys.path.append('.')

from train_all_models import AdaptiveTrainer
from data_manager import load_data

def main():
    """Retrain with optimized profitability settings"""
    
    print("🚀 Starting OPTIMIZED PROFITABILITY Training...")
    print("🎯 Target: Increase profit while maintaining stability")
    print("="*60)
    
    # Load data
    print("📊 Loading EURUSD data...")
    data = load_data('EURUSD')
    
    if data is None or len(data) < 1000:
        print("❌ Failed to load sufficient data")
        return
    
    print(f"✅ Data loaded: {len(data)} records")
    
    # Initialize trainer
    trainer = AdaptiveTrainer('EURUSD')
    
    # ใช้ method ใหม่สำหรับ optimized profitability
    # แก้ไขใน train_all_models.py ก่อนรันสคริปต์นี้
    
    # Start async training with optimized settings
    best_result = trainer.adaptive_train_async(
        data, 
        max_attempts=20,  # ลดลงเพื่อประหยัดเวลา
        target_tier='silver',  # ตั้งเป้า silver tier
        batch_size=3  # 3 models per batch
    )
    
    print("\\n" + "="*60)
    print("🏁 OPTIMIZED PROFITABILITY TRAINING COMPLETE")
    print("="*60)
    
    if best_result:
        print(f"🏆 Best Model Performance:")
        print(f"   📊 Score: {best_result['score']:.1f}")
        print(f"   🥇 Tier: {best_result['tier'].upper()}")
        print(f"   💰 Profit Factor: {best_result['metrics']['profit_factor']:.2f}")
        print(f"   📈 Win Rate: {best_result['metrics']['win_rate']:.1%}")
        print(f"   💵 Total Return: {best_result['metrics']['total_return']:.2%}")
        print(f"   📉 Max Drawdown: {best_result['metrics']['max_drawdown']:.2%}")
    else:
        print("❌ No successful models found")
    
    print("\\n💡 Next Steps:")
    print("   1. Review training logs in training_logs/async_logs/")
    print("   2. Test best model with mt5_trading_bot.py")
    print("   3. Compare with previous bronze model performance")

if __name__ == "__main__":
    main()
'''
    
    with open('quick_retrain_optimized.py', 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print("✅ Created: quick_retrain_optimized.py")

def create_profitability_analysis():
    """สร้างเครื่องมือวิเคราะห์ profitability"""
    
    analysis_content = '''#!/usr/bin/env python3
"""
📊 Profitability Analysis Tool
วิเคราะห์ความสามารถในการทำกำไรของ models
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def analyze_profitability_trends():
    """วิเคราะห์แนวโน้มการทำกำไร"""
    
    # อ่านข้อมูลจากไฟล์ training logs
    files = [
        'training_logs/async_logs/eurusd_async_training.json',
        'training_logs/async_logs/xauusd_async_training.json'
    ]
    
    all_results = []
    
    for file in files:
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                
            for batch in data:
                if 'results' in batch:
                    for result in batch['results']:
                        if result.get('success', False):
                            result['symbol'] = file.split('/')[-1].split('_')[0].upper()
                            all_results.append(result)
        except Exception as e:
            print(f"⚠️ Could not read {file}: {e}")
    
    if not all_results:
        print("❌ No training results found")
        return
    
    df = pd.DataFrame(all_results)
    
    print("📊 PROFITABILITY ANALYSIS")
    print("="*50)
    
    # วิเคราะห์ตาม tier
    print("\\n🏆 Performance by Tier:")
    tier_analysis = df.groupby('tier').agg({
        'score': ['count', 'mean', 'max'],
        'total_return': 'mean',
        'profit_factor': 'mean',
        'win_rate': 'mean'
    }).round(4)
    print(tier_analysis)
    
    # วิเคราะห์ hyperparameters ที่ทำกำไรได้
    print("\\n💰 High Profit Models (>1% return):")
    high_profit = df[df['total_return'] > 0.01]
    if len(high_profit) > 0:
        for idx, model in high_profit.iterrows():
            print(f"   🎯 Score: {model['score']:.1f}, Return: {model['total_return']:.2%}")
            print(f"      📈 LR: {model['hyperparameters']['learning_rate']}")
            print(f"      🎲 Ent Coef: {model['hyperparameters']['ent_coef']}")
            print(f"      📊 Algorithm: {model['hyperparameters']['algorithm']}")
            print()
    else:
        print("   ❌ No models with >1% return found")
    
    # เปรียบเทียบ ent_coef กับ profitability
    print("\\n🎲 Entropy Coefficient vs Profitability:")
    ent_groups = df.groupby(pd.cut(df['ent_coef'], bins=[0, 0.02, 0.05, 1.0, 10.0])).agg({
        'total_return': 'mean',
        'profit_factor': 'mean',
        'score': 'mean'
    }).round(4)
    print(ent_groups)
    
    return df

if __name__ == "__main__":
    df = analyze_profitability_trends()
'''
    
    with open('analyze_profitability.py', 'w', encoding='utf-8') as f:
        f.write(analysis_content)
    
    print("✅ Created: analyze_profitability.py")

def main():
    """Main function"""
    print("🚀 OPTIMIZING PROFITABILITY HYPERPARAMETERS")
    print("="*60)
    
    # สร้าง optimized configs
    configs = create_optimized_profitability_config()
    
    print(f"✅ Created {len(configs)} optimized configurations:")
    for i, config in enumerate(configs, 1):
        print(f"   {i}. {config['name']}")
        print(f"      💡 {config['expected_improvement']}")
        
        # แสดง key parameters
        hp = config['hyperparameters']
        print(f"      📈 LR: {hp['learning_rate']}, Ent: {hp['ent_coef']}")
        if 'clip_range' in hp:
            print(f"      🎯 Clip: {hp['clip_range']}, Steps: {hp['n_steps']}")
        print()
    
    # บันทึก configs
    with open('optimized_profitability_configs.json', 'w') as f:
        json.dump(configs, f, indent=2)
    print("✅ Saved: optimized_profitability_configs.json")
    
    # สร้างสคริปต์ช่วยเหลือ
    create_quick_retrain_script()
    create_profitability_analysis()
    
    print("\n🎯 RECOMMENDATIONS:")
    print("="*40)
    print("1. 🔄 Manual Update Approach:")
    print("   - เปิดไฟล์ train_all_models.py")
    print("   - ไปที่ function generate_hyperparameters()")
    print("   - เปลี่ยน ent_coef จาก [0.01] เป็น [0.02, 0.03, 0.04, 0.05]")
    print("   - เปลี่ยน clip_range จาก [0.15] เป็น [0.17, 0.18, 0.2]")
    print("   - เพิ่ม learning_rate options: [0.0004, 0.0005, 0.0007]")
    print()
    print("2. 🚀 Run Training:")
    print("   python quick_retrain_optimized.py")
    print()
    print("3. 📊 Analyze Results:")
    print("   python analyze_profitability.py")
    print()
    print("💡 Expected Improvements:")
    print("   - เพิ่ม exploration ขึ้น 2-5 เท่า")
    print("   - เพิ่ม total return จาก 0.45% เป็น 2-5%")
    print("   - คงความปลอดภัย (max drawdown < 5%)")
    print("   - เป้าหมาย: Silver tier (score 75+)")

if __name__ == "__main__":
    main()
