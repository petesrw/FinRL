"""
วิเคราะห์แนวโน้มการเรียนรู้และประสิทธิภาพการเทรนหลังแก้ reward
"""

import json
from datetime import datetime
import os

print('📊 การวิเคราะห์แนวโน้มการเรียนรู้หลังแก้ reward function')
print('='*75)

# อ่านข้อมูลการเทรน
async_log_file = "training_logs/async_logs/xauusd_async_training.json"

try:
    with open(async_log_file, 'r') as f:
        training_data = json.load(f)
except FileNotFoundError:
    print(f"❌ ไม่พบไฟล์: {async_log_file}")
    exit(1)

print(f'🔍 วิเคราะห์ข้อมูลจาก: {async_log_file}')
print(f'📈 พบข้อมูล {len(training_data)} batches')
print()

# จัดเรียงข้อมูลตามเวลา
training_data.sort(key=lambda x: x['timestamp'])

# วิเคราะห์แนวโน้มตามเวลา
batch_analysis = []
all_models_chronological = []

print('🕒 การวิเคราะห์แนวโน้มตามลำดับเวลา:')
print('='*60)

for i, batch in enumerate(training_data):
    batch_num = batch['batch_number']
    timestamp = datetime.fromisoformat(batch['timestamp'])
    
    batch_scores = []
    batch_win_rates = []
    batch_profit_factors = []
    batch_returns = []
    
    for result in batch['results']:
        if result['success']:
            score = result['score']
            metrics = result['metrics']
            
            batch_scores.append(score)
            batch_win_rates.append(metrics['win_rate'])
            batch_profit_factors.append(metrics['profit_factor'])
            batch_returns.append(metrics['total_return'])
            
            # เก็บข้อมูลแต่ละ model ตามลำดับเวลา
            all_models_chronological.append({
                'chronological_order': len(all_models_chronological) + 1,
                'batch': batch_num,
                'timestamp': timestamp,
                'score': score,
                'win_rate': metrics['win_rate'],
                'profit_factor': metrics['profit_factor'],
                'total_return': metrics['total_return'],
                'max_drawdown': metrics['max_drawdown']
            })
    
    if batch_scores:
        avg_score = sum(batch_scores) / len(batch_scores)
        max_score = max(batch_scores)
        avg_win_rate = sum(batch_win_rates) / len(batch_win_rates)
        avg_profit_factor = sum(batch_profit_factors) / len(batch_profit_factors)
        profitable_models = len([pf for pf in batch_profit_factors if pf > 1.0])
        
        batch_analysis.append({
            'batch': batch_num,
            'timestamp': timestamp,
            'chronological_index': i + 1,
            'avg_score': avg_score,
            'max_score': max_score,
            'avg_win_rate': avg_win_rate,
            'avg_profit_factor': avg_profit_factor,
            'profitable_models': profitable_models,
            'total_models': len(batch_scores)
        })
        
        print(f'📅 Batch {batch_num} ({timestamp.strftime("%m-%d %H:%M")}):')
        print(f'   📊 คะแนนเฉลี่ย: {avg_score:.1f} | สูงสุด: {max_score:.1f}')
        print(f'   🎯 Win Rate เฉลี่ย: {avg_win_rate:.1%}')
        print(f'   💰 Profit Factor เฉลี่ย: {avg_profit_factor:.2f}')
        print(f'   🏆 Models ที่กำไร: {profitable_models}/{len(batch_scores)}')
        print()

# วิเคราะห์แนวโน้มการปรับปรุง
print('📈 การวิเคราะห์แนวโน้มการเรียนรู้:')
print('='*50)

if len(batch_analysis) >= 3:
    # แบ่งข้อมูลเป็น 3 ช่วง: ต้น กลาง ท้าย
    early_batches = batch_analysis[:len(batch_analysis)//3] or batch_analysis[:1]
    middle_batches = batch_analysis[len(batch_analysis)//3:2*len(batch_analysis)//3] or batch_analysis[1:2]
    late_batches = batch_analysis[2*len(batch_analysis)//3:] or batch_analysis[-1:]
    
    def calculate_period_stats(batches, period_name):
        if not batches:
            return None
            
        avg_scores = [b['avg_score'] for b in batches]
        max_scores = [b['max_score'] for b in batches]
        avg_win_rates = [b['avg_win_rate'] for b in batches]
        avg_profit_factors = [b['avg_profit_factor'] for b in batches]
        
        return {
            'period': period_name,
            'avg_score': sum(avg_scores) / len(avg_scores),
            'best_score': max(max_scores),
            'avg_win_rate': sum(avg_win_rates) / len(avg_win_rates),
            'avg_profit_factor': sum(avg_profit_factors) / len(avg_profit_factors)
        }
    
    early_stats = calculate_period_stats(early_batches, 'ช่วงแรก')
    middle_stats = calculate_period_stats(middle_batches, 'ช่วงกลาง')  
    late_stats = calculate_period_stats(late_batches, 'ช่วงท้าย')
    
    periods = [early_stats, middle_stats, late_stats]
    
    print('🔍 เปรียบเทียบประสิทธิภาพตามช่วงเวลา:')
    print()
    print('   ช่วง    | คะแนนเฉลี่ย | คะแนนสูงสุด | Win Rate | Profit Factor')
    print('   --------|-------------|-------------|----------|---------------')
    
    for stats in periods:
        if stats:
            print(f'   {stats["period"]:>7} | {stats["avg_score"]:>11.1f} | {stats["best_score"]:>11.1f} | {stats["avg_win_rate"]:>8.1%} | {stats["avg_profit_factor"]:>13.2f}')
    
    print()
    
    # วิเคราะห์การปรับปรุง
    if all(periods):
        print('📊 การปรับปรุงประสิทธิภาพ:')
        
        score_trend = late_stats['avg_score'] - early_stats['avg_score']
        win_rate_trend = late_stats['avg_win_rate'] - early_stats['avg_win_rate']
        profit_factor_trend = late_stats['avg_profit_factor'] - early_stats['avg_profit_factor']
        best_score_trend = late_stats['best_score'] - early_stats['best_score']
        
        print(f'   📈 คะแนนเฉลี่ย: {score_trend:+.1f} คะแนน')
        print(f'   📈 คะแนนสูงสุด: {best_score_trend:+.1f} คะแนน')
        print(f'   📈 Win Rate: {win_rate_trend:+.1%}')
        print(f'   📈 Profit Factor: {profit_factor_trend:+.2f}')
        print()
        
        # สรุปแนวโน้ม
        improvements = 0
        if score_trend > 0:
            improvements += 1
        if win_rate_trend > 0:
            improvements += 1
        if profit_factor_trend > 0:
            improvements += 1
        if best_score_trend > 0:
            improvements += 1
            
        print('🎯 สรุปแนวโน้มการเรียนรู้:')
        if improvements >= 3:
            print('   ✅ มีการเรียนรู้ที่ดี - ประสิทธิภาพดีขึ้นในหลายด้าน')
        elif improvements >= 2:
            print('   🤔 มีการเรียนรู้บางส่วน - ปรับปรุงได้ในบางด้าน')
        else:
            print('   ⚠️ การเรียนรู้ยังไม่ชัดเจน - ต้องปรับ hyperparameters')

print()

# วิเคราะห์ความสม่ำเสมอ (Consistency)
print('📊 การวิเคราะห์ความสม่ำเสมอของผลลัพธ์:')
print('='*50)

all_scores = [model['score'] for model in all_models_chronological]
all_win_rates = [model['win_rate'] for model in all_models_chronological]
all_profit_factors = [model['profit_factor'] for model in all_models_chronological]

if len(all_scores) > 1:
    # คำนวณ coefficient of variation (CV) - ยิ่งต่ำยิ่งสม่ำเสมอ
    def coefficient_variation(data):
        if not data or len(data) < 2:
            return 0
        mean = sum(data) / len(data)
        if mean == 0:
            return float('inf')
        variance = sum((x - mean) ** 2 for x in data) / (len(data) - 1)
        std_dev = variance ** 0.5
        return std_dev / mean
    
    score_cv = coefficient_variation(all_scores)
    win_rate_cv = coefficient_variation([wr for wr in all_win_rates if wr > 0])
    profit_factor_cv = coefficient_variation([pf for pf in all_profit_factors if pf > 0])
    
    print(f'📏 ความแปรปรวนสัมพัทธ์ (CV):')
    print(f'   📊 Score CV: {score_cv:.2f} (ยิ่งต่ำยิ่งสม่ำเสมอ)')
    print(f'   🎯 Win Rate CV: {win_rate_cv:.2f}')
    print(f'   💰 Profit Factor CV: {profit_factor_cv:.2f}')
    print()
    
    consistency_score = 0
    if score_cv < 0.3:
        consistency_score += 1
        print('   ✅ Score มีความสม่ำเสมอดี')
    else:
        print('   ⚠️ Score ยังมีความผันแปรสูง')
    
    if win_rate_cv < 0.5:
        consistency_score += 1
        print('   ✅ Win Rate มีความสม่ำเสมอดี')
    else:
        print('   ⚠️ Win Rate ยังมีความผันแปรสูง')
        
    if profit_factor_cv < 1.0:
        consistency_score += 1
        print('   ✅ Profit Factor มีความสม่ำเสมอดี')
    else:
        print('   ⚠️ Profit Factor ยังมีความผันแปรสูง')

print()

# วิเคราะห์จุดเด่นและจุดอ่อน
print('🔍 การวิเคราะห์จุดเด่นและจุดอ่อน:')
print('='*45)

# หา models ที่ดีที่สุด
top_models = sorted(all_models_chronological, key=lambda x: x['score'], reverse=True)[:5]
print('🏆 Top 5 Models ที่ดีที่สุด:')
for i, model in enumerate(top_models, 1):
    print(f'   {i}. Order #{model["chronological_order"]} (Batch {model["batch"]}):')
    print(f'      📊 Score: {model["score"]:.1f} | Win Rate: {model["win_rate"]:.1%}')
    print(f'      💰 PF: {model["profit_factor"]:.2f} | Return: {model["total_return"]:.2%}')

print()

# วิเคราะห์การปรับปรุงตามลำดับเวลา
print('📈 การวิเคราะห์การปรับปรุงตามลำดับการเทรน:')

# แบ่งเป็น 4 ช่วง
total_models = len(all_models_chronological)
quarter_size = max(1, total_models // 4)

quarters = [
    all_models_chronological[:quarter_size],
    all_models_chronological[quarter_size:2*quarter_size],
    all_models_chronological[2*quarter_size:3*quarter_size],
    all_models_chronological[3*quarter_size:]
]

quarter_names = ['Q1 (เริ่มต้น)', 'Q2', 'Q3', 'Q4 (ล่าสุด)']

print('   Quarter  | Avg Score | Best Score | Avg Win Rate | Models > PF 1.0')
print('   ---------|-----------|------------|--------------|----------------')

quarter_stats = []
for i, (quarter_data, name) in enumerate(zip(quarters, quarter_names)):
    if quarter_data:
        avg_score = sum(m['score'] for m in quarter_data) / len(quarter_data)
        best_score = max(m['score'] for m in quarter_data)
        avg_win_rate = sum(m['win_rate'] for m in quarter_data) / len(quarter_data)
        profitable = len([m for m in quarter_data if m['profit_factor'] > 1.0])
        
        quarter_stats.append({
            'quarter': i+1,
            'avg_score': avg_score,
            'best_score': best_score,
            'avg_win_rate': avg_win_rate,
            'profitable_count': profitable
        })
        
        print(f'   {name:>8} | {avg_score:>9.1f} | {best_score:>10.1f} | {avg_win_rate:>12.1%} | {profitable}/{len(quarter_data)}')

print()

# สรุปผลการวิเคราะห์
print('🎯 สรุปผลการวิเคราะห์การเรียนรู้:')
print('='*50)

if len(quarter_stats) >= 2:
    first_quarter = quarter_stats[0]
    last_quarter = quarter_stats[-1]
    
    score_improvement = last_quarter['avg_score'] - first_quarter['avg_score']
    win_rate_improvement = last_quarter['avg_win_rate'] - first_quarter['avg_win_rate']
    
    print('📊 การเปลี่ยนแปลงจาก Q1 ไป Q4:')
    print(f'   📈 คะแนนเฉลี่ย: {score_improvement:+.1f} คะแนน')
    print(f'   📈 Win Rate เฉลี่ย: {win_rate_improvement:+.1%}')
    print()
    
    if score_improvement > 2 and win_rate_improvement > 0.05:
        print('🎉 สรุป: มีการเรียนรู้และปรับปรุงที่ชัดเจน!')
        print('   ✅ ทั้งคะแนนและ Win Rate ดีขึ้นอย่างมีนัยสำคัญ')
    elif score_improvement > 0 or win_rate_improvement > 0:
        print('📈 สรุป: มีการเรียนรู้ในระดับปานกลาง')
        print('   🤔 มีการปรับปรุงบางส่วน แต่ยังต้องปรับแต่งเพิ่มเติม')
    else:
        print('⚠️ สรุป: ยังไม่เห็นการเรียนรู้ที่ชัดเจน')
        print('   🔧 ต้องปรับ reward function หรือ hyperparameters')

print()
print('💡 คำแนะนำสำหรับการปรับปรุงต่อไป:')

best_score = max(model['score'] for model in all_models_chronological)
avg_profit_factor = sum(model['profit_factor'] for model in all_models_chronological) / len(all_models_chronological)

if best_score < 50:
    print('   1. เพิ่ม timesteps เพื่อให้ model เรียนรู้นานขึ้น')
    print('   2. ปรับ reward function ให้เน้น consistency มากขึ้น')

if avg_profit_factor < 1.0:
    print('   3. ปรับ transaction cost ให้เหมาะสมกับสภาพตลาด')
    print('   4. ปรับ risk management parameters')

avg_win_rate = sum(model['win_rate'] for model in all_models_chronological) / len(all_models_chronological)
if avg_win_rate < 0.4:
    print('   5. ปรับ learning rate หรือ exploration parameters')
    print('   6. ใช้ ensemble method เพื่อปรับปรุงความแม่นยำ')

print('   7. ใช้ early stopping เพื่อป้องกัน overfitting')
print('   8. ทดลองใช้ different algorithms (SAC, TD3) สำหรับเปรียบเทียบ')
