# Multi-Account MT5 Setup with Separate Installations

This guide shows you how to set up **4 separate MT5 installations** for each trading symbol, providing complete isolation between accounts.

## 🏗️ Setup Options

### Option 1: Portable Installations (Recommended)
**Best for:** Complete isolation, easy management
```
D:/MT5_Portable/
├── EURUSD/
│   ├── terminal64.exe
│   ├── MQL5/
│   └── config/
├── GBPUSD/
│   ├── terminal64.exe
│   ├── MQL5/
│   └── config/
├── USDJPY/
│   ├── terminal64.exe
│   ├── MQL5/
│   └── config/
└── XAUUSD/
    ├── terminal64.exe
    ├── MQL5/
    └── config/
```

### Option 2: Numbered Accounts (Simple)
**Best for:** Easy organization
```
D:/MetaTrader_Accounts/
├── MetaTraderAccount1/ (EURUSD)
├── MetaTraderAccount2/ (GBPUSD)
├── MetaTraderAccount3/ (USDJPY)
└── MetaTraderAccount4/ (XAUUSD)
```

### Option 3: Config-Based (Advanced)
**Best for:** Resource efficiency
- Single MT5 installation with separate config directories
- Uses launch parameters to specify different configs

## 🚀 Quick Setup

### Step 1: Run the Path Setup Tool
```bash
python setup_mt5_paths.py
```

Choose option 2 for portable installations (recommended).

### Step 2: Configure Account Credentials
```bash
python configure_mt5_accounts.py
```

### Step 3: Test the Setup
```bash
python test_multi_account.py
```

### Step 4: Start Trading
```bash
python multi_symbol_launcher.py
```

## 📋 Manual Setup Example

If you prefer manual setup, update your `.env` file:

```properties
# EURUSD Account
MT5_LOGIN_EURUSD=272015671
MT5_PASSWORD_EURUSD=P@ssw0rd
MT5_SERVER_EURUSD=Exness-MT5Trial14
MT5_PATH_EURUSD=D:/MT5_Portable/EURUSD/terminal64.exe

# GBPUSD Account  
MT5_LOGIN_GBPUSD=272015672
MT5_PASSWORD_GBPUSD=P@ssw0rd
MT5_SERVER_GBPUSD=Exness-MT5Trial15
MT5_PATH_GBPUSD=D:/MT5_Portable/GBPUSD/terminal64.exe

# USDJPY Account
MT5_LOGIN_USDJPY=272015673
MT5_PASSWORD_USDJPY=P@ssw0rd
MT5_SERVER_USDJPY=Exness-MT5Trial16
MT5_PATH_USDJPY=D:/MT5_Portable/USDJPY/terminal64.exe

# XAUUSD Account
MT5_LOGIN_XAUUSD=272015674
MT5_PASSWORD_XAUUSD=P@ssw0rd
MT5_SERVER_XAUUSD=Exness-MT5Trial17
MT5_PATH_XAUUSD=D:/MT5_Portable/XAUUSD/terminal64.exe
```

## 🎯 Benefits of Separate Installations

### Complete Isolation
- ✅ Each symbol has its own MT5 process
- ✅ Independent settings and configurations
- ✅ No interference between accounts
- ✅ Separate chart layouts and expert advisors

### Better Performance
- ✅ Each instance can use different resources
- ✅ Crashes in one don't affect others
- ✅ Independent memory management
- ✅ Parallel processing capabilities

### Enhanced Security
- ✅ Account credentials stored separately
- ✅ Different data directories
- ✅ Isolated connection handling
- ✅ Independent backup strategies

## 🔧 Configuration Details

### Environment Variables Set:
For each symbol, these variables are configured:
- `MT5_LOGIN_{SYMBOL}` - Account login number
- `MT5_PASSWORD_{SYMBOL}` - Account password
- `MT5_SERVER_{SYMBOL}` - Broker server
- `MT5_PATH_{SYMBOL}` - Path to terminal64.exe
- `MT5_DATA_PATH_{SYMBOL}` - Data directory path
- `MT5_CONFIG_PATH_{SYMBOL}` - Configuration directory

### System Requirements
- **RAM**: 4GB+ recommended (1GB per MT5 instance)
- **CPU**: Multi-core processor recommended
- **Storage**: 2GB+ per MT5 installation
- **Network**: Stable internet connection

## 🎮 Usage Examples

### Check All Installations
```python
from setup_mt5_paths import MT5InstallationManager
manager = MT5InstallationManager()
manager.show_current_paths()
```

### Test Specific Symbol
```python
from multi_account_mt5 import multi_mt5
success, config = multi_mt5.get_connection_for_symbol('EURUSD')
print(f"EURUSD MT5 Path: {config['mt5_path']}")
```

### Monitor All Connections
```python
from multi_symbol_launcher import MultiSymbolTradingLauncher
launcher = MultiSymbolTradingLauncher()
launcher.print_status_report()
```

## 🛡️ Safety & Monitoring

### Resource Monitoring
- Monitor CPU usage per MT5 instance
- Watch memory consumption
- Check disk space for logs and data
- Monitor network connections

### Error Handling
- Each instance fails independently
- Automatic reconnection per symbol
- Separate error logs per account
- Graceful shutdown handling

### Backup Strategy
- Backup each MT5 installation separately
- Save configuration files per account
- Export trading history per symbol
- Store logs in separate directories

## 🔧 Troubleshooting

### Installation Issues
1. **Path not found**: Verify MT5 installation exists
2. **Permission denied**: Run as administrator
3. **Copy failed**: Check disk space and permissions

### Connection Issues
1. **Login failed**: Verify account credentials
2. **Server unreachable**: Check internet connection
3. **Multiple instances**: Ensure different accounts

### Performance Issues
1. **High CPU**: Limit concurrent instances
2. **Memory usage**: Increase system RAM
3. **Slow responses**: Check network latency

## 💡 Pro Tips

### Optimal Setup
- Use SSD for MT5 installations
- Assign CPU cores to specific instances
- Use different network adapters if available
- Set up dedicated monitoring

### Account Management
- Use descriptive names for installations
- Document account assignments
- Regular backup of configurations
- Monitor account balances separately

### Trading Optimization
- Different timeframes per symbol
- Symbol-specific trading hours
- Independent risk parameters
- Separate performance metrics

## 📞 Support Commands

### Validate Setup
```bash
python setup_mt5_paths.py  # Choose option 5
```

### Test Connections
```bash
python test_multi_account.py
```

### Configure Accounts
```bash
python configure_mt5_accounts.py
```

### Start Trading
```bash
python multi_symbol_launcher.py
```

---

**🎉 You now have a professional multi-installation MT5 trading system with complete account isolation!**

Each symbol runs in its own MT5 environment with dedicated resources and configurations. This provides maximum stability and performance for your multi-symbol trading strategy.
