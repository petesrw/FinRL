# 🌍 MT5 Cross-Platform Setup Guide
## Windows, macOS, and Linux Installation & Configuration

### 📖 Table of Contents
1. [Quick Setup](#quick-setup)
2. [Windows Installation](#windows-installation)
3. [macOS Installation](#macos-installation-winecrossover)
4. [Linux Installation](#linux-installation-wine)
5. [Path Configuration](#path-configuration)
6. [Troubleshooting](#troubleshooting)
7. [Testing & Verification](#testing--verification)

---

## 🚀 Quick Setup

### Step 1: Run MT5 Detection Utility
```bash
# Automatically detect MT5 installations
python mt5_setup_utility.py
```

This utility will:
- 🔍 Scan your system for MT5 installations
- ✅ Test each installation
- 📝 Generate configuration for your .env file
- 💡 Provide installation recommendations if needed

### Step 2: Update Your .env File
Copy the generated configuration to your `.env` file:
```env
# Example output from utility
MT5_PATH_WINDOWS_CUSTOM=C:\Program Files\MetaTrader 5\terminal64.exe
MT5_AUTO_DETECT_PATH=false
```

---

## 🪟 Windows Installation

### Method 1: Official Installation (Recommended)

1. **Download MT5**:
   - Visit: https://www.metatrader5.com/en/download
   - Download the Windows version

2. **Install MT5**:
   ```bash
   # Run installer as Administrator
   # Choose installation directory (note the path!)
   ```

3. **Common Installation Paths**:
   ```
   C:\Program Files\MetaTrader 5\terminal64.exe
   C:\Program Files (x86)\MetaTrader 5\terminal64.exe
   %LOCALAPPDATA%\Programs\MetaTrader 5\terminal64.exe
   ```

4. **Configure .env**:
   ```env
   # Windows MT5 Configuration
   MT5_PATH_WINDOWS_CUSTOM=C:\Program Files\MetaTrader 5\terminal64.exe
   MT5_DATA_PATH_WINDOWS=%APPDATA%\MetaQuotes\Terminal
   ```

### Method 2: Portable Installation

1. **Download Portable Version**:
   - Some brokers provide portable MT5
   - Extract to desired location (e.g., `D:\MT5\`)

2. **Configure .env**:
   ```env
   # Portable MT5 Configuration
   MT5_PATH_WINDOWS_CUSTOM=D:\MT5\terminal64.exe
   ```

### Windows Configuration Examples:

```env
# Standard Installation
MT5_PATH_WINDOWS_DEFAULT=C:\Program Files\MetaTrader 5\terminal64.exe

# AppData Installation
MT5_PATH_WINDOWS_APPDATA=%LOCALAPPDATA%\Programs\MetaTrader 5\terminal64.exe

# Custom Installation
MT5_PATH_WINDOWS_CUSTOM=D:\MetaTraderAccount1\terminal64.exe

# Data Directory
MT5_DATA_PATH_WINDOWS=%APPDATA%\MetaQuotes\Terminal
```

---

## 🍎 macOS Installation (Wine/CrossOver)

### Method 1: Using Wine (Free)

1. **Install Homebrew** (if not installed):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. **Install Wine**:
   ```bash
   brew install wine-stable
   ```

3. **Download MT5**:
   - Download Windows version of MT5
   - Save to Downloads folder

4. **Install MT5 through Wine**:
   ```bash
   # Navigate to Downloads
   cd ~/Downloads
   
   # Install MT5
   wine MetaTrader5Setup.exe
   
   # Follow installation wizard
   ```

5. **Configure .env**:
   ```env
   # macOS Wine Configuration
   MT5_PATH_MACOS_WINE=~/.wine/drive_c/Program Files/MetaTrader 5/terminal64.exe
   MT5_DATA_PATH_MACOS=~/.wine/drive_c/users/crossover/Application Data/MetaQuotes/Terminal
   ```

### Method 2: Using CrossOver (Paid, More Stable)

1. **Install CrossOver**:
   - Download from: https://www.codeweavers.com/crossover
   - Install the application

2. **Create New Bottle**:
   ```bash
   # Open CrossOver
   # Create new Windows 10 bottle named "MetaTrader5"
   ```

3. **Install MT5**:
   ```bash
   # In CrossOver, select the bottle
   # Run installer: MetaTrader5Setup.exe
   ```

4. **Configure .env**:
   ```env
   # macOS CrossOver Configuration
   MT5_PATH_MACOS_CROSSOVER=~/Library/Application Support/CrossOver/Bottles/MetaTrader5/drive_c/Program Files/MetaTrader 5/terminal64.exe
   ```

### macOS Configuration Examples:

```env
# Wine Installation
MT5_PATH_MACOS_WINE=~/.wine/drive_c/Program Files/MetaTrader 5/terminal64.exe

# CrossOver Installation
MT5_PATH_MACOS_CROSSOVER=~/Library/Application Support/CrossOver/Bottles/MetaTrader5/drive_c/Program Files/MetaTrader 5/terminal64.exe

# Custom Wine Prefix
MT5_PATH_MACOS_CUSTOM=~/wine-prefixes/mt5/drive_c/Program Files/MetaTrader 5/terminal64.exe

# Data Directory
MT5_DATA_PATH_MACOS=~/.wine/drive_c/users/crossover/Application Data/MetaQuotes/Terminal
```

---

## 🐧 Linux Installation (Wine)

### Method 1: Ubuntu/Debian

1. **Install Wine**:
   ```bash
   # Update package list
   sudo apt update
   
   # Install Wine
   sudo apt install wine
   
   # Configure Wine
   winecfg
   ```

2. **Download and Install MT5**:
   ```bash
   # Download MT5 Windows version
   wget https://download.mql5.com/cdn/web/metaquotes.software.corp/mt5/mt5setup.exe
   
   # Install through Wine
   wine mt5setup.exe
   ```

3. **Configure .env**:
   ```env
   # Linux Wine Configuration
   MT5_PATH_LINUX_WINE=~/.wine/drive_c/Program Files/MetaTrader 5/terminal64.exe
   MT5_DATA_PATH_LINUX=~/.wine/drive_c/users/$USER/Application Data/MetaQuotes/Terminal
   ```

### Method 2: Fedora/CentOS

1. **Install Wine**:
   ```bash
   # Fedora
   sudo dnf install wine
   
   # CentOS (enable EPEL first)
   sudo yum install epel-release
   sudo yum install wine
   ```

2. **Install MT5**:
   ```bash
   # Same as Ubuntu method
   wine mt5setup.exe
   ```

### Method 3: Arch Linux

1. **Install Wine**:
   ```bash
   # Install Wine
   sudo pacman -S wine
   
   # Install additional dependencies
   sudo pacman -S wine-gecko wine-mono
   ```

2. **Install MT5**:
   ```bash
   wine mt5setup.exe
   ```

### Method 4: Using PlayOnLinux (GUI)

1. **Install PlayOnLinux**:
   ```bash
   # Ubuntu/Debian
   sudo apt install playonlinux
   
   # Fedora
   sudo dnf install playonlinux
   ```

2. **Install MT5**:
   ```bash
   # Open PlayOnLinux
   # Create new prefix for MT5
   # Install MT5 through the GUI
   ```

3. **Configure .env**:
   ```env
   # PlayOnLinux Configuration
   MT5_PATH_LINUX_PLAYONLINUX=~/.PlayOnLinux/wineprefix/MetaTrader5/drive_c/Program Files/MetaTrader 5/terminal64.exe
   ```

### Linux Configuration Examples:

```env
# Standard Wine Installation
MT5_PATH_LINUX_WINE=~/.wine/drive_c/Program Files/MetaTrader 5/terminal64.exe

# PlayOnLinux Installation
MT5_PATH_LINUX_PLAYONLINUX=~/.PlayOnLinux/wineprefix/MetaTrader5/drive_c/Program Files/MetaTrader 5/terminal64.exe

# Custom Wine Prefix
MT5_PATH_LINUX_CUSTOM=~/wine-prefixes/mt5/drive_c/Program Files/MetaTrader 5/terminal64.exe

# Data Directory
MT5_DATA_PATH_LINUX=~/.wine/drive_c/users/$USER/Application Data/MetaQuotes/Terminal
```

---

## ⚙️ Path Configuration

### Automatic Detection (Recommended)

```env
# Enable automatic path detection
MT5_AUTO_DETECT_PATH=true
MT5_TIMEOUT_SECONDS=30
```

The system will automatically detect MT5 installations in common locations.

### Manual Configuration

```env
# Disable auto-detection
MT5_AUTO_DETECT_PATH=false

# Specify exact path based on your OS
# Windows
MT5_PATH_WINDOWS_CUSTOM=C:\Your\Custom\Path\terminal64.exe

# macOS
MT5_PATH_MACOS_CUSTOM=~/your/custom/path/terminal64.exe

# Linux
MT5_PATH_LINUX_CUSTOM=~/your/custom/path/terminal64.exe
```

### Environment Variables

You can use environment variables in paths:

```env
# Windows
MT5_PATH_WINDOWS_CUSTOM=%USERPROFILE%\MT5\terminal64.exe

# macOS/Linux
MT5_PATH_MACOS_CUSTOM=$HOME/MT5/terminal64.exe
MT5_PATH_LINUX_CUSTOM=$HOME/MT5/terminal64.exe
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. MT5 Not Found

**Problem**: `MT5 executable not found`

**Solutions**:
```bash
# Run detection utility
python mt5_setup_utility.py

# Check if MT5 is installed
# Windows
dir "C:\Program Files\MetaTrader 5\"

# macOS/Linux
ls ~/.wine/drive_c/Program\ Files/MetaTrader\ 5/
```

#### 2. Wine Issues (macOS/Linux)

**Problem**: `Wine execution failed`

**Solutions**:
```bash
# Check Wine installation
wine --version

# Configure Wine
winecfg

# Install Wine dependencies
# Ubuntu
sudo apt install wine32 wine64

# macOS
brew install wine-stable
```

#### 3. Permission Issues

**Problem**: `Permission denied`

**Solutions**:
```bash
# Make executable (Linux/macOS)
chmod +x ~/.wine/drive_c/Program\ Files/MetaTrader\ 5/terminal64.exe

# Run as administrator (Windows)
# Right-click → Run as administrator
```

#### 4. Path with Spaces

**Problem**: Paths with spaces not working

**Solutions**:
```env
# Use quotes for paths with spaces
MT5_PATH_WINDOWS_CUSTOM="C:\Program Files\MetaTrader 5\terminal64.exe"

# Or escape spaces
MT5_PATH_LINUX_WINE=~/.wine/drive_c/Program\ Files/MetaTrader\ 5/terminal64.exe
```

### Debug Mode

Enable debug mode for detailed logging:

```env
DEBUG_MODE=true
LOG_LEVEL=DEBUG
```

### System-Specific Troubleshooting

#### Windows
```bash
# Check if MT5 service is running
tasklist | findstr "terminal64"

# Check registry entries
reg query "HKLM\SOFTWARE\MetaQuotes\MetaTrader 5"
```

#### macOS
```bash
# Check Wine processes
ps aux | grep wine

# Check Wine configuration
wine regedit
```

#### Linux
```bash
# Check Wine processes
ps aux | grep wine

# Check Wine prefix
ls ~/.wine/drive_c/
```

---

## ✅ Testing & Verification

### Method 1: Using Detection Utility

```bash
# Run comprehensive test
python mt5_setup_utility.py
```

### Method 2: Manual Testing

```python
# Test MT5 connection
import MetaTrader5 as mt5
from config import get_config

config = get_config()

# Test connection
if mt5.initialize(path=config.mt5.path):
    print("✅ MT5 connection successful")
    account_info = mt5.account_info()
    print(f"Account: {account_info.login}")
    mt5.shutdown()
else:
    print("❌ MT5 connection failed")
    print(f"Error: {mt5.last_error()}")
```

### Method 3: Configuration Test

```bash
# Test configuration loading
python config.py
```

### Verification Checklist

- [ ] MT5 executable found and accessible
- [ ] MT5 can be launched
- [ ] Python MetaTrader5 module can connect
- [ ] Account login works
- [ ] Data directory accessible
- [ ] Configuration loads without errors

---

## 📋 Quick Reference

### Detection Command
```bash
python mt5_setup_utility.py
```

### Common Paths

#### Windows:
```
C:\Program Files\MetaTrader 5\terminal64.exe
%LOCALAPPDATA%\Programs\MetaTrader 5\terminal64.exe
```

#### macOS:
```
~/.wine/drive_c/Program Files/MetaTrader 5/terminal64.exe
~/Library/Application Support/CrossOver/Bottles/*/drive_c/Program Files/MetaTrader 5/terminal64.exe
```

#### Linux:
```
~/.wine/drive_c/Program Files/MetaTrader 5/terminal64.exe
~/.PlayOnLinux/wineprefix/MetaTrader5/drive_c/Program Files/MetaTrader 5/terminal64.exe
```

### Configuration Template

```env
# Auto-detection (recommended)
MT5_AUTO_DETECT_PATH=true

# Manual configuration (if needed)
MT5_PATH_WINDOWS_CUSTOM=your_windows_path
MT5_PATH_MACOS_CUSTOM=your_macos_path
MT5_PATH_LINUX_CUSTOM=your_linux_path

# Connection settings
MT5_LOGIN=your_account_number
MT5_PASSWORD=your_password
MT5_SERVER=your_broker_server
```

---

## 🎯 Next Steps

1. **Run Detection**: `python mt5_setup_utility.py`
2. **Update .env**: Add generated configuration
3. **Test Connection**: `python config.py`
4. **Start Trading**: `python forex_system_with_config.py`

Your MT5 is now configured for cross-platform forex trading! 🚀

---

*Need help? Check the troubleshooting section or run the detection utility for automated assistance.*