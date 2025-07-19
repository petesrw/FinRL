# 🗑️ การลบข้อมูลการเทรนเก่า - รีเซ็ตระบบ

วันที่: 19 กรกฎาคม 2025

## 🔄 เหตุผลการรีเซ็ต:
- ปรับปรุงเกณฑ์การให้คะแนน (Scoring Criteria) แล้ว
- เกณฑ์ Tier ใหม่สมจริงขึ้นสำหรับ Forex Trading
- ต้องเริ่มการเทรนใหม่ด้วยระบบคะแนนที่แก้ไขแล้ว

## 📊 เกณฑ์ใหม่:
- **Bronze**: Win Rate 55%, PF 1.5, DD 20%, Score ≥ 45
- **Silver**: Win Rate 60%, PF 1.8, DD 18%, Score ≥ 55  
- **Gold**: Win Rate 65%, PF 2.2, DD 15%, Score ≥ 70
- **Diamond**: Win Rate 70%, PF 2.5, DD 12%, Score ≥ 85

## 🗂️ ข้อมูลที่ลบออก:
- ✅ `/training_logs/history/` - ประวัติการเทรน
- ✅ `/training_logs/async_logs/` - ประวัติ async training
- ✅ `/training_logs/failed_configs/` - config ที่ล้มเหลว
- ✅ `/training_logs/successful_configs/` - config ที่สำเร็จ
- ✅ `/models/bronze/`, `/models/silver/`, `/models/gold/`, `/models/diamond/` - model ทุก tier
- ✅ `xauusd_training_history.json` - ประวัติการเทรนหลัก
- ✅ `xauusd_*_results_*.json/txt` - ผลลัพธ์การเทรน
- ✅ `__pycache__/` และ `*.pyc` - cache files

## 📈 ข้อมูลที่เก็บไว้:
- `xauusd_professional_history.json` - ข้อมูลราคาดิบ
- `train_data/` - ข้อมูลการเทรน
- `async_config.py`, `train_all_models.py` - code ที่ปรับปรุงแล้ว

## 🚀 พร้อมเริ่มการเทรนใหม่!

ใช้คำสั่ง:
```bash
python train_all_models.py
```

หรือ async training:
```bash
python -c "
import asyncio
from train_all_models import AdaptiveTrainer
from data_manager import load_training_data

data = load_training_data('XAUUSD')  
trainer = AdaptiveTrainer('XAUUSD')
asyncio.run(trainer.adaptive_train_async(data, max_attempts=50, target_tier='gold', batch_size=4))
"
```
