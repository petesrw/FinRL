# ✅ FinRL Quick Start Checklist

## 🚀 30-Minute Setup Guide

### Step 1: Test Setup (2 minutes)
```bash
python quick_start.py
```
**Expected**: ✅ Success message with AAPL data

---

### Step 2: First AI Trading Bot (10 minutes)
```bash
python simple_trading_example.py
```
**Expected**: 📈 Shows profit/loss results

---

### Step 3: Try Different Stocks (5 minutes)
1. Open `simple_trading_example.py`
2. Change line: `ticker_list = ["AAPL", "MSFT", "GOOGL"]`
3. To: `ticker_list = ["TSLA", "NVDA", "AMD"]`
4. Run again: `python simple_trading_example.py`

---

### Step 4: Full System (15 minutes)
```bash
# Train on 30 stocks (takes 15-30 minutes)
python finrl/main.py --mode=train

# Test the results
python finrl/main.py --mode=test
```

---

## 🎯 Quick Customizations

### Change Stocks:
```python
# Tech stocks
ticker_list = ["TSLA", "NVDA", "AMD"]

# Bank stocks  
ticker_list = ["JPM", "BAC", "WFC"]

# Popular stocks
ticker_list = ["AMZN", "META", "NFLX"]
```

### Change Training Time:
```python
total_timesteps = 50000  # More training (default: 10000)
```

### Change AI Algorithm:
```python
model = agent.get_model("sac")  # Better performance
model = agent.get_model("a2c")  # Faster training
model = agent.get_model("ppo")  # Default, balanced
```

---

## 📊 Understanding Results

### Good Results:
- ✅ Total Return: 8-15% per year
- ✅ Positive numbers
- ✅ Sharpe Ratio > 1.0

### Need Improvement:
- ❌ Negative returns
- ❌ Very high drawdown (>30%)
- ❌ Low win rate (<40%)

---

## 🚨 Common Issues

| Problem | Solution |
|---------|----------|
| "ModuleNotFoundError" | `pip install -r requirements.txt` |
| No data downloaded | Check internet, try different stocks |
| Training too slow | Reduce timesteps, use fewer stocks |
| AI loses money | Try different time periods/algorithms |

---

## 🎯 What to Do Next

### Beginner (Week 1):
- [ ] Run all examples successfully
- [ ] Try 3 different stock combinations  
- [ ] Understand the results

### Intermediate (Week 2-4):
- [ ] Try different AI algorithms
- [ ] Experiment with time periods
- [ ] Compare different sectors

### Advanced (Month 2+):
- [ ] Set up paper trading
- [ ] Create custom strategies
- [ ] Join FinRL community

---

## 📚 Key Files

- `COMPLETE_BEGINNER_GUIDE.md` - Detailed tutorial
- `simple_trading_example.py` - Your first AI bot
- `quick_start.py` - Test if everything works
- `finrl/main.py` - Full system with 30 stocks

---

## ⚠️ Remember

- 🚨 **Never use real money until you're confident**
- 📊 **Start with paper trading**
- 🎯 **Begin with 3-5 stocks, not 30**
- 📈 **Good AI takes time to develop**
- 🤖 **Even good AI can lose money sometimes**

---

**Ready to start? Run:** `python quick_start.py` 🚀