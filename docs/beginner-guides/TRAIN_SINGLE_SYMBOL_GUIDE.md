# 📚 คู่มือการใช้งาน train_single_symbol.py

## 🎯 ไฟล์นี้คืออะไร?

`train_single_symbol.py` เป็นโปรแกรมสำหรับฝึกสอน AI ให้เทรด Forex โดยใช้เทคนิค **Reinforcement Learning (RL)** 

**เปรียบเทียบง่ายๆ**: เหมือนการสอนเด็กเล่นเกม โดยให้รางวัลเมื่อทำถูก และลงโทษเมื่อทำผิด จนเด็กเรียนรู้วิธีเล่นเกมให้เก่ง

---

## 🏗️ โครงสร้างของโปรแกรม

### 1. **SimpleForexEnv Class** - สภาพแวดล้อมการเทรด

```python
class SimpleForexEnv(gym.Env):
```

**คืออะไร**: เป็นการจำลองตลาด Forex ให้ AI ได้ฝึกเทรด

**ทำหน้าที่**:
- จำลองการซื้อ-ขาย
- คำนวณกำไร-ขาดทุน
- ให้รางวัลหรือลงโทษ AI

---

## 🔧 Parameters สำคัญ

### **การตั้งค่าสภาพแวดล้อม**

```python
def __init__(self, data, symbol='XAUUSD', initial_balance=10000, lookback_window=50):
```

| Parameter | คืออะไร | ตัวอย่าง | อธิบาย |
|-----------|---------|----------|--------|
| `data` | ข้อมูลราคา | DataFrame | ข้อมูลราคา OHLC ที่จะใช้ฝึก |
| `symbol` | สัญลักษณ์เทรด | 'XAUUSD' | คู่เงินที่จะเทรด (ทอง/ดอลลาร์) |
| `initial_balance` | เงินเริ่มต้น | 10000 | เงินทุนเริ่มต้น (ดอลลาร์) |
| `lookback_window` | หน้าต่างมองย้อน | 50 | จำนวนแท่งเทียนที่ AI จะดู |

**อธิบาย lookback_window**:
- หาก = 50 หมายความว่า AI จะดูข้อมูล 50 แท่งเทียนล่าสุดก่อนตัดสินใจ
- เหมือนคนเทรดดูกราฟย้อนหลัง 50 แท่งก่อนตัดสินใจซื้อ-ขาย

---

## 📊 Technical Indicators ที่ใช้

### **1. Simple Moving Average (SMA)**
```python
self.data['sma_20'] = self.data['close'].rolling(window=20).mean()
self.data['sma_50'] = self.data['close'].rolling(window=50).mean()
```

**คืออะไร**: ราคาเฉลี่ยเคลื่อนที่
- `SMA_20`: ราคาเฉลี่ย 20 แท่งล่าสุด
- `SMA_50`: ราคาเฉลี่ย 50 แท่งล่าสุด

**ใช้ทำไม**: ดูแนวโน้มราคา
- ราคาเหนือ SMA = แนวโน้มขึ้น
- ราคาใต้ SMA = แนวโน้มลง

### **2. RSI (Relative Strength Index)**
```python
delta = self.data['close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
self.data['rsi'] = 100 - (100 / (1 + rs))
```

**คืออะไร**: ตัวชี้วัดความแข็งแกร่งของราคา (0-100)

**การอ่าน**:
- RSI > 70 = ราคาสูงเกินไป (Overbought) → อาจจะลง
- RSI < 30 = ราคาต่ำเกินไป (Oversold) → อาจจะขึ้น
- RSI 30-70 = ปกติ

### **3. MACD (Moving Average Convergence Divergence)**
```python
ema_12 = self.data['close'].ewm(span=12).mean()
ema_26 = self.data['close'].ewm(span=26).mean()
self.data['macd'] = ema_12 - ema_26
self.data['macd_signal'] = self.data['macd'].ewm(span=9).mean()
```

**คืออะไร**: ตัวชี้วัดโมเมนตัม
- `MACD Line`: EMA12 - EMA26
- `Signal Line`: EMA9 ของ MACD

**การอ่าน**:
- MACD ข้าม Signal ขึ้น = สัญญาณซื้อ
- MACD ข้าม Signal ลง = สัญญาณขาย

---

## 🎮 Action Space - การกระทำของ AI

```python
self.action_space = spaces.Discrete(3)
```

AI มี 3 การกระทำ:

| Action | ค่า | ความหมาย | เมื่อไหร่ใช้ |
|--------|-----|-----------|-------------|
| Hold | 0 | รอดู/ไม่ทำอะไร | เมื่อไม่แน่ใจ |
| Buy | 1 | ซื้อ (Long) | เมื่อคิดว่าราคาจะขึ้น |
| Sell | 2 | ขาย (Short) | เมื่อมีโพซิชั่น Long แล้ว |

**หมายเหตุ**: โปรแกรมนี้เป็น Simple Version ที่ทำได้แค่ Long เท่านั้น

---

## 🏆 Reward System - ระบบให้รางวัล

```python
if profit > 0:
    reward = 1          # ได้กำไร = รางวัล +1
    self.profitable_trades += 1
else:
    reward = -1         # ขาดทุน = โทษ -1
```

**วิธีการ**:
- AI ได้กำไร → ได้รางวัล (+1)
- AI ขาดทุน → ถูกลงโทษ (-1)
- AI เรียนรู้จากรางวัล/โทษนี้

---

## 🤖 PPO Model Parameters

### **การสร้างโมเดล**
```python
model = PPO(
    "MlpPolicy",           # ประเภท Policy
    env,                   # สภาพแวดล้อม
    verbose=1,             # แสดงผลการเทรน
    learning_rate=0.0003,  # อัตราการเรียนรู้
    n_steps=2048,          # จำนวน Steps ต่อ Update
    batch_size=64,         # ขนาด Batch
    n_epochs=10,           # จำนวน Epochs
    gamma=0.99,            # Discount Factor
    gae_lambda=0.95,       # GAE Lambda
    clip_range=0.2,        # Clipping Range
    ent_coef=0.01          # Entropy Coefficient
)
```

### **อธิบาย Parameters**

| Parameter | คืออะไร | ค่าที่ใช้ | ทำไมใช้ค่านี้ |
|-----------|---------|----------|---------------|
| `learning_rate` | อัตราการเรียนรู้ | 0.0003 | ไม่เร็วเกินไป ไม่ช้าเกินไป |
| `n_steps` | Steps ต่อการอัพเดท | 2048 | เก็บข้อมูลพอสำหรับการเรียนรู้ |
| `batch_size` | ขนาดกลุ่มข้อมูล | 64 | ประมวลผลได้เร็ว ไม่กิน RAM มาก |
| `n_epochs` | รอบการเรียนรู้ | 10 | เรียนรู้ข้อมูลหลายรอบ |
| `gamma` | ค่าลด Future Reward | 0.99 | ให้ความสำคัญกับอนาคตใกล้ |
| `gae_lambda` | GAE Parameter | 0.95 | ปรับสมดุล Bias-Variance |
| `clip_range` | ช่วง Clipping | 0.2 | ป้องกันการเปลี่ยนแปลงมากเกินไป |
| `ent_coef` | Entropy Coefficient | 0.01 | ส่งเสริมการสำรวจ |

---

## 🚀 วิธีการใช้งาน

### **1. เตรียมข้อมูล**
```bash
# ต้องมีไฟล์ข้อมูลใน folder train_data/
train_data/XAUUSD/XAUUSD_M5_real.csv
```

### **2. รันโปรแกรม**
```bash
python train_single_symbol.py
```

### **3. ปรับแต่ง Parameters**
```python
def main():
    symbol = 'XAUUSD'      # เปลี่ยนคู่เงิน
    timesteps = 100000     # เปลี่ยนจำนวนการเทรน
    
    model_path = train_model(symbol, timesteps)
```

---

## 📈 การอ่านผลลัพธ์

### **ระหว่างการเทรน**
```
🔄 Training XAUUSD model...
📊 Loading data from train_data/XAUUSD/XAUUSD_M5_real.csv
   Loaded 50,000 rows
🏗️ Creating trading environment...
🤖 Creating PPO model...
🚀 Starting training for 100,000 timesteps...
```

### **ผลลัพธ์การทดสอบ**
```
📈 Test Results:
   Initial balance: $10,000.00    # เงินเริ่มต้น
   Final balance: $12,500.00      # เงินสุดท้าย
   Total return: 25.00%           # ผลตอบแทนรวม
   Total trades: 45               # จำนวนการเทรดทั้งหมด
   Win rate: 67.0%                # อัตราชนะ
   Total reward: 156.50           # รางวัลรวม
```

### **การแปลผล**

| ตัวชี้วัด | ดี | ปานกลาง | ต้องปรับปรุง |
|----------|---|---------|-------------|
| Total Return | > 20% | 5-20% | < 5% |
| Win Rate | > 60% | 50-60% | < 50% |
| Total Trades | 20-100 | 10-20 | < 10 |

---

## ⚙️ การปรับแต่งเพื่อผลลัพธ์ที่ดีขึ้น

### **1. เพิ่มจำนวนการเทรน**
```python
timesteps = 500000  # เพิ่มจาก 100,000
```

### **2. ปรับ Learning Rate**
```python
learning_rate=0.0001  # ลดลงเพื่อเรียนรู้ละเอียดขึ้น
# หรือ
learning_rate=0.001   # เพิ่มขึ้นเพื่อเรียนรู้เร็วขึ้น
```

### **3. เพิ่ม Lookback Window**
```python
lookback_window=100   # ให้ AI ดูข้อมูลย้อนหลังมากขึ้น
```

### **4. ปรับ Batch Size**
```python
batch_size=128        # เพิ่มขึ้นถ้ามี RAM เพียงพอ
```

---

## 🔍 การแก้ปัญหาที่พบบ่อย

### **1. ไฟล์ข้อมูลไม่พบ**
```
❌ Data file not found: train_data/XAUUSD/XAUUSD_M5_real.csv
```
**วิธีแก้**: ตรวจสอบว่ามีไฟล์ข้อมูลในตำแหน่งที่ถูกต้อง

### **2. AI ไม่เทรด (Total trades = 0)**
**สาเหตุ**: 
- ข้อมูลไม่เพียงพอ
- Reward system ไม่เหมาะสม

**วิธีแก้**:
- เพิ่มจำนวน timesteps
- ปรับ learning_rate

### **3. Win Rate ต่ำมาก (< 30%)**
**วิธีแก้**:
- เพิ่ม lookback_window
- ปรับ learning_rate ให้ต่ำลง
- เพิ่มจำนวนการเทรน

### **4. AI เทรดมากเกินไป**
**วิธีแก้**:
- เพิ่ม transaction cost
- ปรับ reward system

---

## 📝 ไฟล์ที่สร้างขึ้น

### **1. Model File**
```
models/xauusd_ppo_20250117_143022.zip
```
**คืออะไร**: โมเดล AI ที่เทรนแล้ว สามารถนำไปใช้เทรดจริงได้

### **2. Results File**
```
xauusd_training_results_20250117_143022.txt
```
**คืออะไร**: ผลลัพธ์การเทรนแบบละเอียด

---

## 🎓 สรุป

`train_single_symbol.py` เป็นเครื่องมือสำหรับ:

1. **ฝึกสอน AI** ให้เทรด Forex
2. **ทดสอบกลยุทธ์** การเทรดแบบต่างๆ
3. **เรียนรู้** Reinforcement Learning

**ข้อดี**:
- ใช้งานง่าย
- ปรับแต่งได้
- ผลลัพธ์ชัดเจน

**ข้อจำกัด**:
- เทรดได้แค่ Long
- Indicators จำกัด
- ไม่มี Risk Management

**เหมาะสำหรับ**: ผู้เริ่มต้นที่ต้องการเรียนรู้ AI Trading

---

## 🔗 ขั้นตอนถัดไป

หลังจากเข้าใจไฟล์นี้แล้ว แนะนำให้ศึกษา:

1. `train_all_models.py` - การเทรนแบบ Advanced
2. `forex_trading_system.py` - ระบบเทรดจริง
3. Technical Analysis เพิ่มเติม

**Happy Learning! 🚀**