# 🔄 RL Training Workflow - ขั้นตอนการเทรน AI

## 🎯 ภาพรวมของกระบวนการ

การเทรน AI สำหรับเทรด Forex ประกอบด้วย 6 ขั้นตอนหลัก:

```
1. เตรียมข้อมูล → 2. สร้างสภาพแวดล้อม → 3. กำหนด Parameters
        ↓                    ↓                      ↓
6. ประเมินผล ← 5. ทดสอบโมเดล ← 4. เทรนโมเดล
```

---

## 📊 ขั้นตอนที่ 1: เตรียมข้อมูล

### **1.1 ข้อมูลที่ต้องการ**
```csv
timestamp,open,high,low,close,volume
2024-01-01 00:00:00,2000.50,2001.20,1999.80,2000.90,1500
2024-01-01 00:05:00,2000.90,2002.10,2000.40,2001.50,1200
```

**คอลัมน์ที่จำเป็น**:
- `timestamp`: เวลา
- `open`: ราคาเปิด
- `high`: ราคาสูงสุด
- `low`: ราคาต่ำสุด
- `close`: ราคาปิด
- `volume`: ปริมาณ (ไม่บังคับ)

### **1.2 การตรวจสอบข้อมูล**
```python
# โหลดข้อมูล
df = pd.read_csv('XAUUSD_M5_real.csv')

# ตรวจสอบข้อมูล
print(f"จำนวนแถว: {len(df):,}")
print(f"ช่วงเวลา: {df['timestamp'].min()} ถึง {df['timestamp'].max()}")
print(f"ข้อมูลหาย: {df.isnull().sum().sum()}")

# ตรวจสอบความต่อเนื่อง
df['timestamp'] = pd.to_datetime(df['timestamp'])
time_diff = df['timestamp'].diff().mode()[0]
print(f"ช่วงเวลาระหว่างข้อมูล: {time_diff}")
```

### **1.3 การทำความสะอาดข้อมูล**
```python
# ลบข้อมูลที่ผิดปกติ
df = df.dropna()  # ลบข้อมูลหาย
df = df[df['high'] >= df['low']]  # ตรวจสอบ high >= low
df = df[df['close'] > 0]  # ตรวจสอบราคาเป็นบวก

# เรียงลำดับตามเวลา
df = df.sort_values('timestamp').reset_index(drop=True)
```

---

## 🏗️ ขั้นตอนที่ 2: สร้างสภาพแวดล้อม

### **2.1 การคำนวณ Technical Indicators**
```python
def _calculate_indicators(self):
    # Moving Averages
    self.data['sma_20'] = self.data['close'].rolling(window=20).mean()
    self.data['sma_50'] = self.data['close'].rolling(window=50).mean()
    
    # RSI
    delta = self.data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    self.data['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_12 = self.data['close'].ewm(span=12).mean()
    ema_26 = self.data['close'].ewm(span=26).mean()
    self.data['macd'] = ema_12 - ema_26
    self.data['macd_signal'] = self.data['macd'].ewm(span=9).mean()
```

### **2.2 การกำหนด Action Space**
```python
# 3 การกระทำ: Hold, Buy, Sell
self.action_space = spaces.Discrete(3)

# หรือ 4 การกระทำ: Hold, Buy, Sell, Close
self.action_space = spaces.Discrete(4)
```

### **2.3 การกำหนด Observation Space**
```python
# ข้อมูลที่ AI จะเห็น
features = ['open', 'high', 'low', 'close', 'sma_20', 'sma_50', 'rsi', 'macd']
lookback_window = 50

self.observation_space = spaces.Box(
    low=-np.inf, 
    high=np.inf, 
    shape=(lookback_window, len(features)), 
    dtype=np.float32
)
```

---

## ⚙️ ขั้นตอนที่ 3: กำหนด Parameters

### **3.1 Environment Parameters**
```python
# การตั้งค่าสภาพแวดล้อม
initial_balance = 10000      # เงินเริ่มต้น
lookback_window = 50         # จำนวนแท่งเทียนที่ดู
transaction_cost = 0.0001    # ค่าธรรมเนียม
max_position_size = 1.0      # ขนาดโพซิชั่นสูงสุด
```

### **3.2 Model Parameters**
```python
# การตั้งค่าโมเดล PPO
model_params = {
    'learning_rate': 0.0003,
    'n_steps': 2048,
    'batch_size': 64,
    'n_epochs': 10,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'ent_coef': 0.01
}
```

### **3.3 Training Parameters**
```python
# การตั้งค่าการเทรน
total_timesteps = 100000     # จำนวนการเทรนทั้งหมด
eval_freq = 10000           # ความถี่ในการประเมิน
save_freq = 50000           # ความถี่ในการบันทึก
```

---

## 🚀 ขั้นตอนที่ 4: เทรนโมเดล

### **4.1 การสร้างโมเดล**
```python
from stable_baselines3 import PPO

# สร้างสภาพแวดล้อม
env = SimpleForexEnv(data, symbol='XAUUSD')
env = DummyVecEnv([lambda: env])

# สร้างโมเดล
model = PPO(
    "MlpPolicy",
    env,
    **model_params,
    verbose=1
)
```

### **4.2 การเทรน**
```python
# เริ่มเทรน
print("🚀 เริ่มการเทรน...")
start_time = datetime.now()

model.learn(total_timesteps=total_timesteps)

end_time = datetime.now()
training_time = end_time - start_time
print(f"⏱️ เทรนเสร็จใน {training_time}")
```

### **4.3 การติดตามความคืบหน้า**
```python
# ใช้ Callback เพื่อติดตามผล
from stable_baselines3.common.callbacks import EvalCallback

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path='./best_model/',
    log_path='./logs/',
    eval_freq=eval_freq,
    deterministic=True,
    render=False
)

model.learn(
    total_timesteps=total_timesteps,
    callback=eval_callback
)
```

---

## 🧪 ขั้นตอนที่ 5: ทดสอบโมเดล

### **5.1 การแบ่งข้อมูล**
```python
# แบ่งข้อมูล 80:20
train_size = int(len(df) * 0.8)
train_data = df[:train_size]
test_data = df[train_size:]

print(f"ข้อมูลเทรน: {len(train_data):,} แถว")
print(f"ข้อมูลทดสอบ: {len(test_data):,} แถว")
```

### **5.2 การทดสอบ**
```python
# สร้างสภาพแวดล้อมทดสอบ
test_env = SimpleForexEnv(test_data, symbol='XAUUSD')

# ทดสอบโมเดล
obs, _ = test_env.reset()
total_reward = 0
done = False
actions_taken = []

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, info = test_env.step(action)
    total_reward += reward
    actions_taken.append(action)
```

### **5.3 การบันทึกผลการทดสอบ**
```python
# เก็บผลลัพธ์
results = {
    'final_balance': info['balance'],
    'total_return': (info['balance'] - test_env.initial_balance) / test_env.initial_balance,
    'total_trades': info['total_trades'],
    'profitable_trades': info['profitable_trades'],
    'win_rate': info['profitable_trades'] / max(info['total_trades'], 1),
    'total_reward': total_reward,
    'actions_distribution': {
        'hold': actions_taken.count(0),
        'buy': actions_taken.count(1),
        'sell': actions_taken.count(2)
    }
}
```

---

## 📈 ขั้นตอนที่ 6: ประเมินผล

### **6.1 ตัวชี้วัดหลัก**
```python
def calculate_metrics(results, test_env):
    metrics = {}
    
    # ผลตอบแทน
    metrics['total_return'] = results['total_return']
    metrics['annualized_return'] = results['total_return'] * (252/len(test_data)) * 24 * 12  # สำหรับ M5
    
    # ความเสี่ยง
    equity_curve = test_env.equity_curve
    returns = np.diff(equity_curve) / equity_curve[:-1]
    metrics['volatility'] = np.std(returns) * np.sqrt(252 * 24 * 12)
    metrics['sharpe_ratio'] = metrics['annualized_return'] / metrics['volatility']
    
    # Drawdown
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (peak - equity_curve) / peak
    metrics['max_drawdown'] = np.max(drawdown)
    
    # การเทรด
    metrics['win_rate'] = results['win_rate']
    metrics['profit_factor'] = calculate_profit_factor(test_env.trades)
    
    return metrics
```

### **6.2 การแสดงผล**
```python
def print_results(metrics):
    print("📊 ผลการประเมิน")
    print("="*50)
    print(f"📈 Total Return: {metrics['total_return']:.2%}")
    print(f"📅 Annualized Return: {metrics['annualized_return']:.2%}")
    print(f"📊 Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"📉 Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"🎯 Win Rate: {metrics['win_rate']:.1%}")
    print(f"💰 Profit Factor: {metrics['profit_factor']:.2f}")
```

### **6.3 การสร้างกราฟ**
```python
import matplotlib.pyplot as plt

def plot_results(test_env):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Equity Curve
    axes[0,0].plot(test_env.equity_curve)
    axes[0,0].set_title('Equity Curve')
    axes[0,0].set_ylabel('Balance ($)')
    
    # Drawdown
    equity = np.array(test_env.equity_curve)
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak * 100
    axes[0,1].fill_between(range(len(drawdown)), drawdown, alpha=0.3, color='red')
    axes[0,1].set_title('Drawdown')
    axes[0,1].set_ylabel('Drawdown (%)')
    
    # Actions Distribution
    actions = [trade['action'] for trade in test_env.trades]
    action_counts = [actions.count(i) for i in range(3)]
    axes[1,0].bar(['Hold', 'Buy', 'Sell'], action_counts)
    axes[1,0].set_title('Actions Distribution')
    
    # Monthly Returns
    monthly_returns = calculate_monthly_returns(test_env.equity_curve)
    axes[1,1].bar(range(len(monthly_returns)), monthly_returns)
    axes[1,1].set_title('Monthly Returns')
    axes[1,1].set_ylabel('Return (%)')
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()
```

---

## 🔄 การปรับปรุงโมเดล

### **7.1 การวิเคราะห์ผลลัพธ์**
```python
def analyze_performance(metrics):
    issues = []
    suggestions = []
    
    # ตรวจสอบ Win Rate
    if metrics['win_rate'] < 0.5:
        issues.append("Win Rate ต่ำ")
        suggestions.append("ลด learning_rate หรือเพิ่ม lookback_window")
    
    # ตรวจสอบ Sharpe Ratio
    if metrics['sharpe_ratio'] < 1.0:
        issues.append("Sharpe Ratio ต่ำ")
        suggestions.append("ปรับ reward function หรือลด transaction_cost")
    
    # ตรวจสอบ Max Drawdown
    if metrics['max_drawdown'] > 0.2:
        issues.append("Max Drawdown สูง")
        suggestions.append("เพิ่ม risk management หรือลด position_size")
    
    return issues, suggestions
```

### **7.2 การปรับแต่ง Parameters**
```python
def optimize_parameters(current_params, metrics):
    new_params = current_params.copy()
    
    # ถ้า Win Rate ต่ำ
    if metrics['win_rate'] < 0.5:
        new_params['learning_rate'] *= 0.5  # ลดครึ่งหนึ่ง
        new_params['n_epochs'] *= 2         # เพิ่มเป็น 2 เท่า
    
    # ถ้า Sharpe Ratio ต่ำ
    if metrics['sharpe_ratio'] < 1.0:
        new_params['ent_coef'] *= 2         # เพิ่มการสำรวจ
    
    # ถ้า Max Drawdown สูง
    if metrics['max_drawdown'] > 0.2:
        new_params['clip_range'] *= 0.5     # ลดการเปลี่ยนแปลง
    
    return new_params
```

---

## 📋 Checklist การเทรน

### **ก่อนเทรน**
- [ ] ตรวจสอบข้อมูลครบถ้วน
- [ ] ทำความสะอาดข้อมูล
- [ ] คำนวณ Technical Indicators
- [ ] แบ่งข้อมูล Train/Test
- [ ] ตั้งค่า Parameters

### **ระหว่างเทรน**
- [ ] ติดตามความคืบหน้า
- [ ] บันทึก Log
- [ ] ตรวจสอบ Memory Usage
- [ ] สำรองโมเดลระหว่างทาง

### **หลังเทรน**
- [ ] ทดสอบโมเดล
- [ ] คำนวณ Metrics
- [ ] สร้างกราฟผลลัพธ์
- [ ] วิเคราะห์ปัญหา
- [ ] บันทึกผลการทดลอง

---

## 🎯 เป้าหมายที่ดี

### **สำหรับผู้เริ่มต้น**
- Win Rate > 55%
- Total Return > 10%
- Max Drawdown < 20%
- Sharpe Ratio > 0.5

### **สำหรับระดับกลาง**
- Win Rate > 60%
- Total Return > 20%
- Max Drawdown < 15%
- Sharpe Ratio > 1.0

### **สำหรับระดับสูง**
- Win Rate > 65%
- Total Return > 30%
- Max Drawdown < 10%
- Sharpe Ratio > 1.5

---

## 🚨 ข้อผิดพลาดที่ควรหลีกเลี่ยง

### **1. Overfitting**
- **อาการ**: ผลดีในการเทรน แต่แย่ในการทดสอบ
- **แก้ไข**: ใช้ Cross-validation, ลด Model Complexity

### **2. Data Leakage**
- **อาการ**: ใช้ข้อมูลอนาคตในการตัดสินใจ
- **แก้ไข**: ตรวจสอบ Feature Engineering

### **3. Insufficient Data**
- **อาการ**: โมเดลไม่เรียนรู้
- **แก้ไข**: เพิ่มข้อมูล หรือลด Model Complexity

### **4. Wrong Reward Function**
- **อาการ**: AI ทำพฤติกรรมแปลกๆ
- **แก้ไข**: ปรับ Reward Function ให้สมเหตุสมผล

---

## 📚 สรุป

การเทรน RL สำหรับ Forex เป็นกระบวนการที่ต้องอาศัย:

1. **ข้อมูลที่ดี**: สะอาด, ครบถ้วน, ต่อเนื่อง
2. **Parameters ที่เหมาะสม**: ปรับแต่งตามข้อมูลและเป้าหมาย
3. **การประเมินผลที่ถูกต้อง**: ใช้ Metrics ที่หลากหลาย
4. **การปรับปรุงอย่างต่อเนื่อง**: วิเคราะห์และปรับแต่ง

**ความสำเร็จมาจากการทดลองและปรับปรุงอย่างต่อเนื่อง!** 🚀