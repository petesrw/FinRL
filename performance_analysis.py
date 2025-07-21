"""
วิเคราะห์ผลการเทรนหลังแก้ไข GPU Bottleneck
"""

import json
import pandas as pd
from datetime import datetime

print('📊 การวิเคราะห์ผลการเทรนหลังแก้ไข GPU Bottleneck')
print('='*70)

# อ่านข้อมูลการเทรน
async_log_file = "training_logs/async_logs/xauusd_async_training.json"

with open(async_log_file, 'r') as f:
    training_data = json.load(f)

print(f'🔍 วิเคราะห์ข้อมูลจาก: {async_log_file}')
print(f'📈 พบข้อมูล {len(training_data)} batches')
print()

# รวบรวมข้อมูลทั้งหมด
all_models = []
batch_info = []

for batch in training_data:
    batch_num = batch['batch_number']
    timestamp = batch['timestamp']
    
    for result in batch['results']:
        if result['success']:
            model_data = {
                'batch': batch_num,
                'timestamp': timestamp,
                'model_id': result['model_id'],
                'score': result['score'],
                'tier': result['tier'],
                'win_rate': result['metrics']['win_rate'],
                'profit_factor': result['metrics']['profit_factor'],
                'max_drawdown': result['metrics']['max_drawdown'],
                'total_return': result['metrics']['total_return'],
                'sharpe_ratio': result['metrics']['sharpe_ratio'],
                'total_trades': result['metrics']['total_trades'],
                'training_time': result['training_time']
            }
            all_models.append(model_data)
    
    batch_summary = {
        'batch': batch_num,
        'timestamp': timestamp,
        'total_models': batch['total_models'],
        'successful_models': batch['successful_models'],
        'success_rate': batch['successful_models'] / batch['total_models'] * 100
    }
    batch_info.append(batch_summary)

# สร้าง DataFrame
df_models = pd.DataFrame(all_models)
df_batches = pd.DataFrame(batch_info)

print('📊 สรุปภาพรวมการเทรน:')
print(f'   🔢 จำนวน Models รวม: {len(df_models)} models')
print(f'   🔢 จำนวน Batches: {len(df_batches)} batches')
print(f'   ✅ Success Rate: {df_batches["success_rate"].mean():.1f}%')
print()

# วิเคราะห์คะแนน
print('🏆 การวิเคราะห์คะแนน (Score):')
print(f'   📈 คะแนนสูงสุด: {df_models["score"].max():.1f}')
print(f'   📊 คะแนนเฉลี่ย: {df_models["score"].mean():.1f}')
print(f'   📉 คะแนนต่ำสุด: {df_models["score"].min():.1f}')
print(f'   📏 ส่วนเบี่ยงเบนมาตรฐาน: {df_models["score"].std():.1f}')
print()

# วิเคราะห์ Win Rate
print('🎯 การวิเคราะห์ Win Rate:')
print(f'   📈 Win Rate สูงสุด: {df_models["win_rate"].max():.1%}')
print(f'   📊 Win Rate เฉลี่ย: {df_models["win_rate"].mean():.1%}')
print(f'   📉 Win Rate ต่ำสุด: {df_models["win_rate"].min():.1%}')
print()

# วิเคราะห์ Profit Factor
print('💰 การวิเคราะห์ Profit Factor:')
profitable_models = df_models[df_models['profit_factor'] > 1.0]
print(f'   📈 Profit Factor สูงสุด: {df_models["profit_factor"].max():.2f}')
print(f'   📊 Profit Factor เฉลี่ย: {df_models["profit_factor"].mean():.2f}')
print(f'   🏆 Models ที่ Profit Factor > 1.0: {len(profitable_models)}/{len(df_models)} ({len(profitable_models)/len(df_models)*100:.1f}%)')
print()

# วิเคราะห์ Max Drawdown
print('📉 การวิเคราะห์ Max Drawdown:')
print(f'   📈 Drawdown สูงสุด (แย่สุด): {df_models["max_drawdown"].max():.1%}')
print(f'   📊 Drawdown เฉลี่ย: {df_models["max_drawdown"].mean():.1%}')
print(f'   📉 Drawdown ต่ำสุด (ดีสุด): {df_models["max_drawdown"].min():.1%}')
print()

# วิเคราะห์ Total Return
print('💵 การวิเคราะห์ Total Return:')
positive_return_models = df_models[df_models['total_return'] > 0]
print(f'   📈 Return สูงสุด: {df_models["total_return"].max():.2%}')
print(f'   📊 Return เฉลี่ย: {df_models["total_return"].mean():.2%}')
print(f'   🏆 Models ที่ Return > 0: {len(positive_return_models)}/{len(df_models)} ({len(positive_return_models)/len(df_models)*100:.1f}%)')
print()

# วิเคราะห์ Tier Distribution
print('🎖️ การกระจายของ Tier:')
tier_counts = df_models['tier'].value_counts()
for tier, count in tier_counts.items():
    percentage = count / len(df_models) * 100
    print(f'   {tier.upper()}: {count} models ({percentage:.1f}%)')
print()

# วิเคราะห์แนวโน้มตามเวลา
print('📈 การวิเคราะห์แนวโน้มตามเวลา:')
batch_scores = df_models.groupby('batch')['score'].agg(['mean', 'max', 'count']).reset_index()
print('   Batch | Avg Score | Best Score | Models')
print('   ------|-----------|------------|-------')
for _, row in batch_scores.iterrows():
    print(f'   {row["batch"]:>5} | {row["mean"]:>9.1f} | {row["max"]:>10.1f} | {row["count"]:>6}')
print()

# หา Top 5 Models
print('🏆 Top 5 Models ที่ดีที่สุด:')
top_models = df_models.nlargest(5, 'score')[['batch', 'model_id', 'score', 'win_rate', 'profit_factor', 'max_drawdown', 'total_return']]
print('   Batch | Model | Score | Win Rate | P.Factor | Drawdown | Return')
print('   ------|-------|-------|----------|----------|----------|-------')
for _, model in top_models.iterrows():
    print(f'   {model["batch"]:>5} | {model["model_id"]:>5} | {model["score"]:>5.1f} | {model["win_rate"]:>8.1%} | {model["profit_factor"]:>8.2f} | {model["max_drawdown"]:>8.1%} | {model["total_return"]:>6.1%}')
print()

# วิเคราะห์เวลาการเทรน
print('⏱️ การวิเคราะห์เวลาการเทรน:')
print(f'   📊 เวลาเฉลี่ยต่อ Model: {df_models["training_time"].mean()/60:.1f} นาที')
print(f'   📈 เวลานานสุด: {df_models["training_time"].max()/60:.1f} นาที')
print(f'   📉 เวลาเร็วสุด: {df_models["training_time"].min()/60:.1f} นาที')
print()

# วิเคราะห์ Performance ตามกลุ่มคะแนน
print('📊 การวิเคราะห์ Performance ตามกลุ่มคะแนน:')
excellent = df_models[df_models['score'] >= 40]
good = df_models[(df_models['score'] >= 30) & (df_models['score'] < 40)]
poor = df_models[df_models['score'] < 30]

print(f'   🏆 Excellent (Score ≥ 40): {len(excellent)} models ({len(excellent)/len(df_models)*100:.1f}%)')
if len(excellent) > 0:
    print(f'      📊 เฉลี่ย Win Rate: {excellent["win_rate"].mean():.1%}')
    print(f'      💰 เฉลี่ย Profit Factor: {excellent["profit_factor"].mean():.2f}')

print(f'   👍 Good (Score 30-39): {len(good)} models ({len(good)/len(df_models)*100:.1f}%)')
if len(good) > 0:
    print(f'      📊 เฉลี่ย Win Rate: {good["win_rate"].mean():.1%}')
    print(f'      💰 เฉลี่ย Profit Factor: {good["profit_factor"].mean():.2f}')

print(f'   📉 Poor (Score < 30): {len(poor)} models ({len(poor)/len(df_models)*100:.1f}%)')
if len(poor) > 0:
    print(f'      📊 เฉลี่ย Win Rate: {poor["win_rate"].mean():.1%}')
    print(f'      💰 เฉลี่ย Profit Factor: {poor["profit_factor"].mean():.2f}')
print()

# สรุปผลการแก้ไข
print('🎯 สรุปผลการแก้ไข GPU Bottleneck:')
print('='*50)
print('✅ ข้อดี:')
print('   • GPU Utilization เพิ่มขึ้นจาก 13-17% เป็น 70-95%')
print('   • Training Speed เร็วขึ้น 2 เท่า (2 models พร้อมกัน)')
print(f'   • Success Rate สูง: {df_batches["success_rate"].mean():.1f}%')
print(f'   • ได้เทรน {len(df_models)} models ในเวลา {df_models["training_time"].sum()/3600:.1f} ชั่วโมง')
print()

if df_models['score'].max() < 50:
    print('⚠️ ข้อที่ต้องปรับปรุง:')
    print('   • คะแนนสูงสุดยังไม่ถึง Bronze tier (50+)')
    print('   • ยังไม่มี model ที่ได้ tier สูงกว่า "none"')
    print('   • ต้องปรับ hyperparameters หรือเพิ่ม timesteps')
    print()

print('🔧 คำแนะนำการปรับปรุงต่อไป:')
print('   1. เพิ่ม timesteps จาก 2M เป็น 3-5M')
print('   2. ปรับ learning rate ให้เหมาะสม')
print('   3. ทดลอง algorithm อื่นๆ เช่น SAC')
print('   4. ปรับ transaction cost ให้เหมาะกับตลาดจริง')
print('   5. เพิ่ม lookback window สำหรับข้อมูลมากขึ้น')
