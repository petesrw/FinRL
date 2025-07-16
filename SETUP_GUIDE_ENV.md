# 🔧 Forex Trading System Setup Guide
## Complete Configuration with .env Files

### 📖 Table of Contents
1. [Quick Setup (5 minutes)](#quick-setup-5-minutes)
2. [Environment Configuration](#environment-configuration)
3. [MT5 Setup](#mt5-setup)
4. [Configuration Options](#configuration-options)
5. [Running the System](#running-the-system)
6. [Monitoring & Notifications](#monitoring--notifications)
7. [Troubleshooting](#troubleshooting)

---

## 🚀 Quick Setup (5 minutes)

### Step 1: Install Dependencies
```bash
# Install all required packages
pip install -r forex_requirements.txt
```

### Step 2: Create Configuration File
```bash
# Copy the example configuration
cp .env.example .env

# Edit with your settings
notepad .env  # Windows
# or
nano .env     # Linux/Mac
```

### Step 3: Basic Configuration
Edit your `.env` file with these minimum settings:

```env
# MT5 Settings (use demo account first!)
MT5_LOGIN=123456789
MT5_PASSWORD=your_demo_password
MT5_SERVER=MetaQuotes-Demo

# Trading Settings
DEFAULT_SYMBOL=EURUSD
RISK_PER_TRADE=0.02
TARGET_WIN_RATE=0.65

# Model Settings
MODEL_TYPE=PPO
TRAINING_TIMESTEPS=50000

# Safety Settings
DEMO_MODE=true
ENABLE_EMERGENCY_STOP=true
```

### Step 4: Test the System
```bash
# Test configuration loading
python config.py

# Run the trading system
python forex_system_with_config.py
```

---

## 🔧 Environment Configuration

### Configuration File Structure

Your `.env` file controls every aspect of the trading system:

```env
# =============================================================================
# MT5 BROKER SETTINGS
# =============================================================================
MT5_LOGIN=123456789                    # Your MT5 account number
MT5_PASSWORD=your_password             # Your MT5 password
MT5_SERVER=MetaQuotes-Demo            # Your broker's server

# =============================================================================
# TRADING PARAMETERS
# =============================================================================
DEFAULT_SYMBOL=EURUSD                 # Primary trading pair
RISK_PER_TRADE=0.02                   # 2% risk per trade
MAX_DRAWDOWN=0.20                     # 20% maximum drawdown
TARGET_WIN_RATE=0.65                  # 65% target win rate

# =============================================================================
# MODEL SETTINGS
# =============================================================================
MODEL_TYPE=PPO                        # RL Algorithm (PPO/SAC/A2C)
TRAINING_TIMESTEPS=100000              # Training duration
INITIAL_BALANCE=10000.0                # Starting balance for training

# =============================================================================
# SAFETY SETTINGS
# =============================================================================
DEMO_MODE=true                         # Start with demo mode!
ENABLE_EMERGENCY_STOP=true             # Enable safety stops
MAX_CONSECUTIVE_LOSSES=5               # Stop after 5 losses in a row
```

### Key Configuration Sections

#### 🏦 **Broker Settings**
```env
# Demo Account (Recommended for testing)
MT5_LOGIN=123456789
MT5_PASSWORD=demo_password
MT5_SERVER=MetaQuotes-Demo

# Live Account (Only after thorough testing!)
# MT5_LOGIN=987654321
# MT5_PASSWORD=live_password
# MT5_SERVER=YourBroker-Live
```

#### 💰 **Risk Management**
```env
RISK_PER_TRADE=0.02                   # 2% of account per trade
MAX_DRAWDOWN=0.20                     # Stop if 20% drawdown
MAX_DAILY_TRADES=10                   # Maximum trades per day
POSITION_SIZING_METHOD=PERCENT_RISK   # How to calculate position size
```

#### 🤖 **AI Model Settings**
```env
MODEL_TYPE=PPO                        # PPO (stable), SAC (advanced), A2C (fast)
TRAINING_TIMESTEPS=100000             # More = better but slower
LOOKBACK_WINDOW=100                   # How much history to consider
```

#### ⏰ **Trading Schedule**
```env
TRADING_START_HOUR=8                  # Start trading at 8 AM
TRADING_END_HOUR=18                   # Stop trading at 6 PM
TRADING_DAYS=0,1,2,3,4               # Monday to Friday (0=Mon, 6=Sun)
TRADING_TIMEZONE=UTC                  # Your timezone
```

#### 🛡️ **Safety Features**
```env
ENABLE_EMERGENCY_STOP=true            # Enable emergency stops
MAX_CONSECUTIVE_LOSSES=5              # Stop after 5 losses
EMERGENCY_STOP_LOSS_AMOUNT=1000.0    # Stop if lose $1000
AVOID_NEWS_TRADING=true               # Avoid high-impact news times
```

---

## 🔗 MT5 Setup

### Download and Install MT5

1. **Download MT5**: [Official MetaTrader 5](https://www.metatrader5.com/en/download)
2. **Install**: Follow the installation wizard
3. **Open Demo Account**: File → Open Account → Demo Account

### Configure MT5 for Automated Trading

1. **Enable Algo Trading**:
   - Tools → Options → Expert Advisors
   - ✅ Allow automated trading
   - ✅ Allow DLL imports
   - ✅ Allow imports of external experts

2. **Get Account Details**:
   ```
   Login: 123456789 (your demo account number)
   Password: demo_password (your demo password)
   Server: MetaQuotes-Demo (or your broker's demo server)
   ```

3. **Test Connection**:
   ```python
   import MetaTrader5 as mt5
   
   # Test MT5 connection
   if mt5.initialize():
       print("✅ MT5 connected successfully")
       account_info = mt5.account_info()
       print(f"Account: {account_info.login}")
       print(f"Balance: ${account_info.balance}")
       mt5.shutdown()
   else:
       print("❌ MT5 connection failed")
   ```

### Update .env with MT5 Credentials

```env
# Replace with your actual demo account details
MT5_LOGIN=123456789
MT5_PASSWORD=your_demo_password
MT5_SERVER=MetaQuotes-Demo
```

---

## ⚙️ Configuration Options

### Trading Strategies

#### **Conservative Setup** (Recommended for beginners)
```env
RISK_PER_TRADE=0.01                   # 1% risk
TARGET_WIN_RATE=0.70                  # 70% target
MAX_DRAWDOWN=0.10                     # 10% max drawdown
POSITION_SIZING_METHOD=PERCENT_RISK
```

#### **Balanced Setup** (Good for most users)
```env
RISK_PER_TRADE=0.02                   # 2% risk
TARGET_WIN_RATE=0.65                  # 65% target
MAX_DRAWDOWN=0.15                     # 15% max drawdown
POSITION_SIZING_METHOD=PERCENT_RISK
```

#### **Aggressive Setup** (For experienced traders)
```env
RISK_PER_TRADE=0.03                   # 3% risk
TARGET_WIN_RATE=0.60                  # 60% target
MAX_DRAWDOWN=0.20                     # 20% max drawdown
POSITION_SIZING_METHOD=KELLY
```

### Multi-Currency Trading

```env
# Enable multiple currency pairs
ENABLE_MULTI_SYMBOL=true
TRADING_SYMBOLS=EURUSD,GBPUSD,USDJPY,AUDUSD

# Portfolio allocation (must sum to 1.0)
PORTFOLIO_ALLOCATION_EURUSD=0.4       # 40% allocation
PORTFOLIO_ALLOCATION_GBPUSD=0.3       # 30% allocation
PORTFOLIO_ALLOCATION_USDJPY=0.2       # 20% allocation
PORTFOLIO_ALLOCATION_AUDUSD=0.1       # 10% allocation
```

### Technical Indicators

```env
# Customize technical analysis
ATR_PERIOD=14                         # ATR period for volatility
SL_ATR_MULTIPLIER=1.5                # Stop loss = 1.5 * ATR
TP_ATR_MULTIPLIER=2.5                # Take profit = 2.5 * ATR

# Moving averages
SMA_FAST=20                          # Fast SMA period
SMA_SLOW=50                          # Slow SMA period
EMA_FAST=12                          # Fast EMA period
EMA_SLOW=26                          # Slow EMA period

# Oscillators
RSI_PERIOD=14                        # RSI period
STOCH_K_PERIOD=14                    # Stochastic %K period
```

---

## 🚀 Running the System

### Method 1: Quick Start (Recommended)

```bash
# 1. Test configuration
python config.py

# 2. Run the complete system
python forex_system_with_config.py
```

### Method 2: Step by Step

```bash
# 1. Test individual components
python -c "from config import get_config; config = get_config(); config.print_summary()"

# 2. Train models for all symbols
python -c "
from forex_system_with_config import ConfigurableForexBot
from config import get_config

config = get_config()
for symbol in config.get_symbols_list():
    bot = ConfigurableForexBot(symbol)
    bot.train_model()
    bot.test_model()
"

# 3. Start live trading
python forex_system_with_config.py
```

### Method 3: Custom Script

Create `my_trading_bot.py`:

```python
from forex_system_with_config import ConfigurableForexBot
from config import get_config

# Load configuration
config = get_config()

# Create bot for EURUSD
bot = ConfigurableForexBot("EURUSD")

# Train if needed
if not bot.load_model():
    print("Training new model...")
    bot.train_model()

# Test model
if bot.test_model():
    print("Model passed tests - starting trading")
    bot.start_live_trading()
    
    # Keep running
    try:
        while True:
            time.sleep(60)
            report = bot.get_performance_report()
            print(f"Win Rate: {report['performance']['win_rate']:.1%}")
    except KeyboardInterrupt:
        bot.stop_live_trading()
        print("Trading stopped")
else:
    print("Model failed tests - need more training")
```

---

## 📱 Monitoring & Notifications

### Telegram Notifications

1. **Create Telegram Bot**:
   - Message @BotFather on Telegram
   - Send `/newbot`
   - Follow instructions to get bot token

2. **Get Chat ID**:
   - Add your bot to a chat
   - Send a message to the bot
   - Visit: `https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates`
   - Find your chat ID in the response

3. **Configure .env**:
   ```env
   TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz
   TELEGRAM_CHAT_ID=123456789
   ```

### Email Notifications

```env
# Gmail example
EMAIL_SMTP_SERVER=smtp.gmail.com
EMAIL_SMTP_PORT=587
EMAIL_USERNAME=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
EMAIL_TO=alerts@yourdomain.com
```

### Database Logging

```env
# Enable trade logging to database
SAVE_TRADES_TO_DB=true
DB_FILE=trades.db
```

View your trades:
```python
import sqlite3
import pandas as pd

# Connect to database
conn = sqlite3.connect('trades.db')

# View recent trades
trades = pd.read_sql_query("SELECT * FROM trades ORDER BY timestamp DESC LIMIT 10", conn)
print(trades)

# View performance summary
performance = pd.read_sql_query("SELECT * FROM performance ORDER BY timestamp DESC LIMIT 5", conn)
print(performance)

conn.close()
```

---

## 🔍 Monitoring Dashboard

### Real-time Performance Monitoring

```python
# Create monitoring script: monitor.py
from forex_system_with_config import ConfigurableForexBot
from config import get_config
import time

config = get_config()
symbols = config.get_symbols_list()

# Create bots
bots = {}
for symbol in symbols:
    bots[symbol] = ConfigurableForexBot(symbol)

print("📊 Real-time Performance Monitor")
print("=" * 50)

while True:
    print(f"\n⏰ {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    for symbol, bot in bots.items():
        report = bot.get_performance_report()
        stats = report['performance']
        
        print(f"{symbol:8} | "
              f"Trades: {stats['total_trades']:3d} | "
              f"Win Rate: {stats['win_rate']:5.1%} | "
              f"P&L: ${stats['total_profit']:8.2f}")
    
    time.sleep(300)  # Update every 5 minutes
```

Run with: `python monitor.py`

---

## 🚨 Troubleshooting

### Common Issues

#### 1. Configuration Loading Errors

**Error**: `FileNotFoundError: .env file not found`
```bash
# Solution: Create .env file
cp .env.example .env
```

**Error**: `ValueError: Configuration validation failed`
```bash
# Solution: Check your .env file for invalid values
python config.py  # This will show specific errors
```

#### 2. MT5 Connection Issues

**Error**: `MT5 initialization failed`
```bash
# Solutions:
# 1. Make sure MT5 is installed and running
# 2. Enable automated trading in MT5 settings
# 3. Check if your account credentials are correct
```

**Error**: `MT5 login failed`
```bash
# Solutions:
# 1. Verify login, password, and server in .env
# 2. Make sure account is active
# 3. Try connecting manually in MT5 first
```

#### 3. Model Training Issues

**Error**: `CUDA out of memory`
```env
# Solution: Reduce training parameters
TRAINING_TIMESTEPS=10000  # Reduce from 100000
LOOKBACK_WINDOW=50        # Reduce from 100
```

**Error**: `Model training very slow`
```env
# Solution: Use faster algorithm
MODEL_TYPE=A2C           # Instead of PPO or SAC
```

#### 4. Performance Issues

**Problem**: Low win rate
```env
# Solutions:
# 1. Increase training time
TRAINING_TIMESTEPS=200000

# 2. Try different algorithm
MODEL_TYPE=SAC

# 3. Adjust risk parameters
RISK_PER_TRADE=0.01
TARGET_WIN_RATE=0.70
```

**Problem**: High drawdown
```env
# Solutions:
# 1. Reduce risk
RISK_PER_TRADE=0.01
MAX_DRAWDOWN=0.10

# 2. Enable stricter safety
ENABLE_EMERGENCY_STOP=true
MAX_CONSECUTIVE_LOSSES=3
```

### Debug Mode

Enable detailed logging:
```env
DEBUG_MODE=true
LOG_LEVEL=DEBUG
SAVE_DEBUG_DATA=true
```

### Configuration Validation

Test your configuration:
```python
from config import get_config

try:
    config = get_config()
    print("✅ Configuration loaded successfully")
    config.print_summary()
except Exception as e:
    print(f"❌ Configuration error: {e}")
```

---

## 📋 Configuration Checklist

### Before First Run:
- [ ] `.env` file created and configured
- [ ] MT5 installed and demo account opened
- [ ] All dependencies installed (`pip install -r forex_requirements.txt`)
- [ ] Configuration validated (`python config.py`)
- [ ] Demo mode enabled (`DEMO_MODE=true`)

### Before Live Trading:
- [ ] Thoroughly tested in demo mode (1+ months)
- [ ] Model achieves target win rate consistently
- [ ] Risk management parameters set appropriately
- [ ] Emergency stops configured
- [ ] Notifications set up (Telegram/Email)
- [ ] Database logging enabled
- [ ] Live account credentials configured
- [ ] `DEMO_MODE=false` only when ready

### Daily Monitoring:
- [ ] Check system is running
- [ ] Review performance metrics
- [ ] Monitor for emergency stops
- [ ] Check notification alerts
- [ ] Review trade database
- [ ] Verify MT5 connection

---

## 🎯 Quick Configuration Templates

### Template 1: Conservative Beginner
```env
# Copy this to your .env file for conservative trading
MT5_LOGIN=your_demo_login
MT5_PASSWORD=your_demo_password
MT5_SERVER=MetaQuotes-Demo

DEFAULT_SYMBOL=EURUSD
RISK_PER_TRADE=0.01
TARGET_WIN_RATE=0.70
MAX_DRAWDOWN=0.10

MODEL_TYPE=PPO
TRAINING_TIMESTEPS=100000

DEMO_MODE=true
ENABLE_EMERGENCY_STOP=true
MAX_CONSECUTIVE_LOSSES=3
```

### Template 2: Balanced Trader
```env
# Copy this to your .env file for balanced trading
MT5_LOGIN=your_demo_login
MT5_PASSWORD=your_demo_password
MT5_SERVER=MetaQuotes-Demo

DEFAULT_SYMBOL=EURUSD
RISK_PER_TRADE=0.02
TARGET_WIN_RATE=0.65
MAX_DRAWDOWN=0.15

MODEL_TYPE=SAC
TRAINING_TIMESTEPS=150000

DEMO_MODE=true
ENABLE_EMERGENCY_STOP=true
MAX_CONSECUTIVE_LOSSES=5
```

### Template 3: Multi-Currency Advanced
```env
# Copy this to your .env file for multi-currency trading
MT5_LOGIN=your_demo_login
MT5_PASSWORD=your_demo_password
MT5_SERVER=MetaQuotes-Demo

ENABLE_MULTI_SYMBOL=true
TRADING_SYMBOLS=EURUSD,GBPUSD,USDJPY
PORTFOLIO_ALLOCATION_EURUSD=0.5
PORTFOLIO_ALLOCATION_GBPUSD=0.3
PORTFOLIO_ALLOCATION_USDJPY=0.2

RISK_PER_TRADE=0.02
TARGET_WIN_RATE=0.65
MAX_DRAWDOWN=0.20

MODEL_TYPE=SAC
TRAINING_TIMESTEPS=200000

DEMO_MODE=true
ENABLE_EMERGENCY_STOP=true
```

---

## 🎉 You're Ready!

Your forex trading system is now fully configured with:

✅ **Professional Configuration Management**
✅ **Secure Credential Storage**
✅ **Flexible Risk Management**
✅ **Multiple Trading Strategies**
✅ **Real-time Notifications**
✅ **Comprehensive Logging**
✅ **Safety Features**
✅ **Multi-Currency Support**

**Next Steps:**
1. Start with demo trading
2. Monitor performance for several weeks
3. Adjust configuration based on results
4. Only go live after consistent success

**Remember**: Always start with `DEMO_MODE=true` and never risk money you can't afford to lose!

---

*Happy Trading! 🚀💰*