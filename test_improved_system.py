"""
สคริปต์ทดสอบการปรับปรุงระบบเทรน
"""

import json
import os
from datetime import datetime

print('🧪 ทดสอบการปรับปรุงระบบเทรน')
print('='*50)

# สร้าง backup log เก่าก่อนเริ่มทดสอบ
if os.path.exists('training_logs/async_logs/xauusd_async_training.json'):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = f'training_logs/async_logs/xauusd_async_training_backup_{timestamp}.json'
    
    with open('training_logs/async_logs/xauusd_async_training.json', 'r') as src:
        with open(backup_path, 'w') as dst:
            dst.write(src.read())
    
    print(f'📁 Backup เก่า log → {backup_path}')

print()
print('📋 การปรับปรุงที่ทำไปแล้ว:')
print('   ✅ 1. ปรับ Reward Function ให้สมดุลและเน้น consistency')
print('   ✅ 2. เพิ่ม Timesteps จาก 2M เป็น 3-5M')
print('   ✅ 3. ปรับ Learning Rate เป็น 0.0002-0.0004 (conservative)')
print('   ✅ 4. เพิ่ม Early Stopping mechanism')
print('   ✅ 5. เพิ่ม SAC algorithm (30% probability)')
print()

print('🎯 เป้าหมายการปรับปรุง:')
print('   📊 คะแนนเฉลี่ย: > 35 (ปัจจุบัน: 28.7)')
print('   🏆 คะแนนสูงสุด: > 50 (Bronze tier, ปัจจุบัน: 43.9)')
print('   💰 Profit Factor เฉลี่ย: > 1.0 (ปัจจุบัน: 0.44)')
print('   🎯 Win Rate เฉลี่ย: > 40% (ปัจจุบัน: 24.8%)')
print('   🏆 Models ที่กำไร: > 20% (ปัจจุบัน: 9.5%)')
print()

print('⚡ การปรับปรุงหลัก:')
print()

print('1️⃣ REWARD FUNCTION:')
print('   🔄 เปลี่ยนจาก extreme rewards → balanced rewards')
print('   📊 เน้น consistency มากกว่า high returns')
print('   🛑 เพิ่ม drawdown control และ risk management')
print('   🎯 เพิ่ม trading frequency management')
print()

print('2️⃣ TRAINING PARAMETERS:')
print('   🕐 Timesteps: 2M → 4M (100% increase)')
print('   🧠 Learning Rate: 0.0005-0.0008 → 0.0002-0.0004')
print('   📊 Batch Size: focus on 1024-2048')
print('   🎮 Gamma: more conservative 0.96-0.99')
print()

print('3️⃣ EARLY STOPPING:')
print('   🛑 Stop training if no validation improvement')
print('   📊 Check every 10% of training progress')
print('   ⏳ Patience limit: 3 validation checks')
print('   💾 Save best model automatically')
print()

print('4️⃣ ALGORITHM DIVERSITY:')
print('   🤖 PPO: 50% (was 80%)')
print('   🌊 SAC: 30% (new addition)')
print('   🔧 Others: 20%')
print()

print('🧪 วิธีทดสอบ:')
print('   1. รัน train_all_models.py ด้วยการตั้งค่าใหม่')
print('   2. ติดตามผล 5-10 models แรก')
print('   3. เปรียบเทียบกับผลเก่า')
print('   4. ปรับแต่งเพิ่มเติมถ้าจำเป็น')
print()

print('📊 สัญญาณความสำเร็จ:')
print('   ✅ คะแนนเฉลี่ย > 35 ใน 10 models แรก')
print('   ✅ มี model ที่คะแนน > 50 (Bronze tier)')
print('   ✅ Profit Factor > 1.0 ใน > 20% ของ models')
print('   ✅ Win Rate เฉลี่ย > 40%')
print('   ✅ Early stopping ทำงานได้ (รายงาน validation scores)')
print()

print('⚠️ สัญญาณที่ต้องปรับเพิ่ม:')
print('   📉 คะแนนเฉลี่ยไม่ดีขึ้น')
print('   💸 Profit Factor ยังคงต่ำ')
print('   🔄 Models ยัง overfit (validation score ลดลง)')
print('   ⏰ Training time ยาวเกินไป (> 2 ชั่วโมงต่อ model)')
print()

print('🚀 พร้อมเริ่มการทดสอบ!')
print('💡 คำแนะนำ: เริ่มด้วย batch size เล็ก (2 models) เพื่อทดสอบ')
print('📝 บันทึกผลทุก 2-3 models เพื่อติดตามแนวโน้ม')

print()
print('=' * 50)
print('🔧 เริ่มการทดสอบได้แล้ว! รัน: python train_all_models.py')
