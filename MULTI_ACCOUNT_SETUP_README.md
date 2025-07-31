# Multi-Account MT5 Trading System

This system allows you to trade different forex symbols using separate MT5 accounts for better risk isolation and management.

## 🏦 Account Configuration

You now have **4 separate MT5 account paths** for different symbols:

### Current Configuration Template:
```properties
# Account 1: EURUSD Trading
MT5_LOGIN_EURUSD=272015671
MT5_PASSWORD_EURUSD= P@ssw0rd
MT5_SERVER_EURUSD=Exness-MT5Trial14

# Account 2: GBPUSD Trading  
MT5_LOGIN_GBPUSD=272015672
MT5_PASSWORD_GBPUSD= P@ssw0rd
MT5_SERVER_GBPUSD=Exness-MT5Trial15

# Account 3: USDJPY Trading
MT5_LOGIN_USDJPY=272015673
MT5_PASSWORD_USDJPY= P@ssw0rd
MT5_SERVER_USDJPY=Exness-MT5Trial16

# Account 4: XAUUSD (Gold) Trading
MT5_LOGIN_XAUUSD=272015674
MT5_PASSWORD_XAUUSD= P@ssw0rd
MT5_SERVER_XAUUSD=Exness-MT5Trial17
```

## 🚀 Quick Start

### 1. Configure Your Accounts

**Option A: Interactive Setup**
```bash
python configure_mt5_accounts.py
```

**Option B: Manual .env Edit**
Edit your `.env` file and add the account details for each symbol.

**Option C: Use Same Account for All (Testing)**
```bash
python configure_mt5_accounts.py
# Choose option 4 to copy main account to all symbols
```

### 2. Test Your Setup
```bash
python test_multi_account.py
```

### 3. Start Multi-Symbol Trading
```bash
python multi_symbol_launcher.py
```

## 🎯 Benefits of Multi-Account Trading

### Risk Isolation
- Each symbol trades on separate account
- Losses in one symbol don't affect others
- Better portfolio management

### Independent Position Management  
- EURUSD: 0.3% risk, max 2 positions
- GBPUSD: 0.35% risk, max 2 positions
- USDJPY: 0.25% risk, max 1 position
- XAUUSD: 0.4% risk, max 3 positions

### Separate Performance Tracking
- Individual P&L per symbol
- Symbol-specific win rates
- Independent risk metrics

## 📁 New Files Created

### Core System Files
- `multi_account_mt5.py` - Multi-account MT5 interface
- `multi_symbol_launcher.py` - Main trading launcher
- `configure_mt5_accounts.py` - Account configuration helper
- `test_multi_account.py` - System test suite

### Configuration
- Updated `.env` file with multi-account settings

## 🔧 Configuration Options

### Symbol-Specific Settings
Each symbol can have different:
- Risk percentage per trade
- Maximum concurrent positions  
- Trading intervals
- Stop loss / Take profit ratios

### Account Mapping
```python
# In multi_symbol_launcher.py
symbol_configs = {
    'EURUSD': {'risk_percent': 0.3, 'max_positions': 2},
    'GBPUSD': {'risk_percent': 0.35, 'max_positions': 2}, 
    'USDJPY': {'risk_percent': 0.25, 'max_positions': 1},
    'XAUUSD': {'risk_percent': 0.4, 'max_positions': 3}
}
```

## 🎮 Usage Examples

### Start All Symbols
```bash
python multi_symbol_launcher.py
```

### Test Single Symbol
```python
from multi_account_mt5 import multi_mt5

# Test EURUSD connection
success, config = multi_mt5.get_connection_for_symbol('EURUSD')
if success:
    print(f"Connected to account {config['login']}")
```

### Check System Status
```python
from multi_symbol_launcher import MultiSymbolTradingLauncher

launcher = MultiSymbolTradingLauncher()
launcher.print_status_report()
```

## 🛡️ Safety Features

### Connection Management
- Automatic reconnection on failures
- Thread-safe connection handling
- Graceful shutdown on Ctrl+C

### Risk Controls
- Symbol-specific position limits
- Independent stop loss cooldowns
- Account balance monitoring

### Error Handling
- Robust error recovery
- Detailed logging per symbol
- Fallback to main account if needed

## 📊 Monitoring

### Real-time Status
- Connection status per symbol
- Active positions per account
- P&L tracking per symbol

### Logging
- Separate logs per symbol
- Trade execution details
- Error tracking and recovery

## 🔧 Troubleshooting

### Connection Issues
1. Run `test_multi_account.py` to diagnose
2. Check account credentials in `.env`
3. Verify MT5 servers are correct

### Account Setup Issues
1. Use `configure_mt5_accounts.py` for guided setup
2. Ensure login numbers are unique
3. Check server names match your broker

### Trading Issues
1. Verify account balances
2. Check symbol availability on each account
3. Monitor connection stability

## 💡 Tips for Success

### Account Management
- Use different account numbers for true isolation
- Or use same account with different magic numbers
- Monitor margin usage across accounts

### Symbol Selection
- Major pairs (EUR/USD, GBP/USD) for stability
- JPY pairs for different market sessions
- Gold (XAU/USD) for commodity exposure

### Risk Management
- Start with lower risk percentages
- Monitor correlation between symbols
- Use appropriate position sizing

## 🎯 Next Steps

1. **Setup**: Configure your 4 MT5 accounts
2. **Test**: Run the test suite to verify everything works
3. **Trade**: Start with one symbol, then scale up
4. **Monitor**: Use the status reports to track performance
5. **Optimize**: Adjust risk parameters based on results

## 📞 Support

If you encounter issues:
1. Check the test results first
2. Verify .env configuration
3. Ensure MT5 accounts are active and funded
4. Review the detailed logs for specific errors

---

**🎉 You now have a professional multi-account trading system with proper risk isolation and independent symbol management!**
