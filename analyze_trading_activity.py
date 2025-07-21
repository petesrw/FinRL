"""
วิเคราะห์ปัญหา Trading Activity - Bot ไม่ค่อยเทรด
"""

import json

print('🔍 วิเคราะห์ปัญหา Trading Activity')
print('='*50)

# อ่านข้อมูล successful configs
with open('training_logs/successful_configs/xauusd_successful_configs.json', 'r') as f:
    configs = json.load(f)

print(f'📊 วิเคราะห์จาก {len(configs)} successful models')
print()

# วิเคราะห์ Trading Activity
low_activity_models = []
decent_activity_models = []
good_activity_models = []

for config in configs:
    metrics = config.get('metrics', {})
    total_trades = metrics.get('total_trades', 0)
    total_return = metrics.get('total_return', 0)
    balance = metrics.get('balance', 10000)
    equity = metrics.get('equity', 10000)
    
    # คำนวณ actual ROI
    initial_balance = 10000
    actual_roi = ((balance - initial_balance) / initial_balance) * 100
    
    if total_trades <= 10:
        low_activity_models.append({
            'attempt': config.get('attempt'),
            'trades': total_trades,
            'roi': actual_roi,
            'return': total_return * 100,
            'balance': balance,
            'score': config.get('score'),
            'algorithm': config.get('config', {}).get('algorithm')
        })
    elif total_trades <= 50:
        decent_activity_models.append({
            'attempt': config.get('attempt'),
            'trades': total_trades,
            'roi': actual_roi,
            'return': total_return * 100,
            'balance': balance,
            'score': config.get('score'),
            'algorithm': config.get('config', {}).get('algorithm')
        })
    else:
        good_activity_models.append({
            'attempt': config.get('attempt'),
            'trades': total_trades,
            'roi': actual_roi,
            'return': total_return * 100,
            'balance': balance,
            'score': config.get('score'),
            'algorithm': config.get('config', {}).get('algorithm')
        })

print('📈 สรุปการวิเคราะห์:')
print()
print(f'🔴 Models ที่เทรดน้อย (≤10 trades): {len(low_activity_models)} models')
print(f'🟡 Models ที่เทรดปานกลาง (11-50 trades): {len(decent_activity_models)} models')
print(f'🟢 Models ที่เทรดดี (>50 trades): {len(good_activity_models)} models')
print()

print('🔴 Models ที่เทรดน้อยมาก:')
if low_activity_models:
    print('   Attempt | Trades | ROI   | Balance  | Score | Algorithm')
    print('   --------|--------|-------|----------|-------|----------')
    for model in low_activity_models[:10]:  # Show first 10
        print(f'   {model["attempt"]:>7} | {model["trades"]:>6} | {model["roi"]:>5.2f}% | {model["balance"]:>8.0f} | {model["score"]:>5.1f} | {model["algorithm"]}')
else:
    print('   (ไม่มี)')

print()
print('🟡 Models ที่เทรดปานกลาง:')
if decent_activity_models:
    print('   Attempt | Trades | ROI   | Balance  | Score | Algorithm')
    print('   --------|--------|-------|----------|-------|----------')
    for model in decent_activity_models[:5]:
        print(f'   {model["attempt"]:>7} | {model["trades"]:>6} | {model["roi"]:>5.2f}% | {model["balance"]:>8.0f} | {model["score"]:>5.1f} | {model["algorithm"]}')
else:
    print('   (ไม่มี)')

print()
print('🟢 Models ที่เทรดดี:')
if good_activity_models:
    print('   Attempt | Trades | ROI   | Balance  | Score | Algorithm')
    print('   --------|--------|-------|----------|-------|----------')
    for model in good_activity_models[:5]:
        print(f'   {model["attempt"]:>7} | {model["trades"]:>6} | {model["roi"]:>5.2f}% | {model["balance"]:>8.0f} | {model["score"]:>5.1f} | {model["algorithm"]}')
else:
    print('   (ไม่มี)')

print()
print('📊 สถิติรวม:')
all_models = low_activity_models + decent_activity_models + good_activity_models
total_trades_all = [m['trades'] for m in all_models]
total_roi_all = [m['roi'] for m in all_models]

if total_trades_all:
    avg_trades = sum(total_trades_all) / len(total_trades_all)
    avg_roi = sum(total_roi_all) / len(total_roi_all)
    
    print(f'   📈 เฉลี่ยจำนวน trades: {avg_trades:.1f}')
    print(f'   💰 เฉลี่ย ROI: {avg_roi:.2f}%')
    print(f'   🚫 Models ที่ ROI ≈ 0%: {len([m for m in all_models if abs(m["roi"]) < 0.1])} models')

print()
print('🔍 สาเหตุที่เป็นไปได้:')
print('   1. 🎯 Action space ไม่เหมาะสม - bot เลือก "hold" มากเกินไป')
print('   2. 🔄 Reward function ไม่ส่งเสริมการเทรด')
print('   3. 📊 Observation space ไม่ดีพอให้ bot ตัดสินใจ')
print('   4. ⚖️ Transaction cost สูงเกินไป')
print('   5. 🧠 Model ยังไม่เรียนรู้ pattern การเทรดที่ดี')

print()
print('💡 การแก้ไขที่แนะนำ:')
print('   1. 🎁 เพิ่ม reward สำหรับการเทรด (แม้ขาดทุนเล็กน้อย)')
print('   2. 📉 ลด transaction cost เพื่อส่งเสริมการเทรด')
print('   3. 🚫 เพิ่ม penalty สำหรับการ hold นานเกินไป')
print('   4. 🎯 ปรับ action space ให้เอื้อต่อการเทรด')
print('   5. 📊 เพิ่ม market volatility indicators')
