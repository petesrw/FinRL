"""
MT5 Installation Path Manager
Helps set up separate MT5 installations for each trading account
"""

import os
import shutil
import winreg
from pathlib import Path
from dotenv import load_dotenv, set_key

class MT5InstallationManager:
    """Manages separate MT5 installations for different accounts"""
    
    def __init__(self, env_file=".env"):
        self.env_file = env_file
        load_dotenv(env_file)
        
        self.symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD']
        self.base_mt5_path = self._find_mt5_installation()
        
    def _find_mt5_installation(self):
        """Find existing MT5 installation"""
        common_paths = [
            r"C:\Program Files\MetaTrader 5\terminal64.exe",
            r"C:\Program Files (x86)\MetaTrader 5\terminal64.exe",
            os.getenv('MT5_PATH_WINDOWS_DEFAULT', ''),
            r"D:\MetaTraderAccount1\terminal64.exe"
        ]
        
        for path in common_paths:
            if path and os.path.exists(path):
                print(f"✅ Found MT5 installation: {path}")
                return path
        
        # Try to find via registry
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\MetaQuotes\MetaTrader 5") as key:
                path = winreg.QueryValueEx(key, "Path")[0]
                terminal_path = os.path.join(path, "terminal64.exe")
                if os.path.exists(terminal_path):
                    print(f"✅ Found MT5 via registry: {terminal_path}")
                    return terminal_path
        except:
            pass
        
        print("⚠️ MT5 installation not found automatically")
        return None
    
    def show_current_paths(self):
        """Show current MT5 path configuration"""
        print("📂 CURRENT MT5 PATH CONFIGURATION")
        print("="*45)
        
        print(f"Base MT5 Installation: {self.base_mt5_path or 'Not found'}")
        print()
        
        for symbol in self.symbols:
            path = os.getenv(f'MT5_PATH_{symbol}')
            data_path = os.getenv(f'MT5_DATA_PATH_{symbol}')
            
            print(f"{symbol}:")
            print(f"   Executable: {path or 'Not set'}")
            print(f"   Data Path: {data_path or 'Not set'}")
            
            if path and os.path.exists(path):
                print(f"   Status: ✅ Exists")
            elif path:
                print(f"   Status: ❌ Path not found")
            else:
                print(f"   Status: ⚠️ Not configured")
            print()
    
    def create_portable_installations(self, base_dir="D:/MT5_Portable"):
        """Create separate portable MT5 installations for each symbol"""
        if not self.base_mt5_path:
            print("❌ Base MT5 installation not found. Please install MT5 first.")
            return False
        
        base_mt5_dir = os.path.dirname(self.base_mt5_path)
        base_dir = Path(base_dir)
        
        print(f"🔧 CREATING PORTABLE MT5 INSTALLATIONS")
        print(f"Source: {base_mt5_dir}")
        print(f"Target: {base_dir}")
        print("="*50)
        
        success_count = 0
        
        for symbol in self.symbols:
            try:
                symbol_dir = base_dir / symbol
                symbol_mt5_path = symbol_dir / "terminal64.exe"
                symbol_data_path = symbol_dir / "MQL5" / "Data"
                symbol_config_path = symbol_dir / "config"
                
                print(f"\n📁 Setting up {symbol}...")
                
                # Create directories
                symbol_dir.mkdir(parents=True, exist_ok=True)
                symbol_data_path.mkdir(parents=True, exist_ok=True)
                symbol_config_path.mkdir(parents=True, exist_ok=True)
                
                # Copy MT5 files if not already copied
                if not symbol_mt5_path.exists():
                    print(f"   📂 Copying MT5 files...")
                    shutil.copytree(base_mt5_dir, symbol_dir, dirs_exist_ok=True)
                    print(f"   ✅ Files copied to {symbol_dir}")
                else:
                    print(f"   ✅ Files already exist in {symbol_dir}")
                
                # Update .env file
                set_key(self.env_file, f'MT5_PATH_{symbol}', str(symbol_mt5_path))
                set_key(self.env_file, f'MT5_DATA_PATH_{symbol}', str(symbol_data_path))
                set_key(self.env_file, f'MT5_CONFIG_PATH_{symbol}', str(symbol_config_path))
                
                print(f"   ✅ Environment configured for {symbol}")
                success_count += 1
                
            except Exception as e:
                print(f"   ❌ Error setting up {symbol}: {e}")
        
        print(f"\n📊 Setup Results: {success_count}/{len(self.symbols)} symbols configured")
        
        if success_count > 0:
            print("\n💡 NEXT STEPS:")
            print("1. Each symbol now has its own MT5 installation")
            print("2. Run test_multi_account.py to verify connections")
            print("3. Start trading with multi_symbol_launcher.py")
            print("\n⚠️ IMPORTANT:")
            print("- Each MT5 instance will run independently")
            print("- Configure each with its specific account credentials")
            print("- Ensure sufficient system resources (RAM/CPU)")
        
        return success_count > 0
    
    def create_config_based_setup(self, base_config_dir="D:/MT5_Configs"):
        """Create separate config directories (same MT5 exe, different configs)"""
        if not self.base_mt5_path:
            print("❌ Base MT5 installation not found. Please install MT5 first.")
            return False
        
        base_config_path = Path(base_config_dir)
        
        print(f"🔧 CREATING CONFIG-BASED MT5 SETUP")
        print(f"MT5 Executable: {self.base_mt5_path}")
        print(f"Config Base: {base_config_path}")
        print("="*45)
        
        success_count = 0
        
        for symbol in self.symbols:
            try:
                config_dir = base_config_path / symbol
                data_dir = config_dir / "Data"
                profiles_dir = config_dir / "Profiles"
                
                print(f"\n📁 Setting up config for {symbol}...")
                
                # Create directories
                config_dir.mkdir(parents=True, exist_ok=True)
                data_dir.mkdir(parents=True, exist_ok=True)
                profiles_dir.mkdir(parents=True, exist_ok=True)
                
                # Create launch command with config parameter
                launch_command = f'"{self.base_mt5_path}" /config:"{config_dir}"'
                
                # Update .env file
                set_key(self.env_file, f'MT5_PATH_{symbol}', launch_command)
                set_key(self.env_file, f'MT5_DATA_PATH_{symbol}', str(data_dir))
                set_key(self.env_file, f'MT5_CONFIG_PATH_{symbol}', str(config_dir))
                
                print(f"   ✅ Config directory created: {config_dir}")
                print(f"   ✅ Launch command: {launch_command}")
                success_count += 1
                
            except Exception as e:
                print(f"   ❌ Error setting up config for {symbol}: {e}")
        
        print(f"\n📊 Setup Results: {success_count}/{len(self.symbols)} configs created")
        return success_count > 0
    
    def create_simple_directory_setup(self, base_dir="D:/MetaTrader_Accounts"):
        """Create separate directories with numbered MT5 accounts"""
        if not self.base_mt5_path:
            print("❌ Base MT5 installation not found. Please install MT5 first.")
            return False
        
        base_mt5_dir = os.path.dirname(self.base_mt5_path)
        base_path = Path(base_dir)
        
        print(f"🔧 CREATING SIMPLE DIRECTORY SETUP")
        print(f"Source: {base_mt5_dir}")
        print(f"Base: {base_path}")
        print("="*40)
        
        success_count = 0
        
        for i, symbol in enumerate(self.symbols, 1):
            try:
                account_dir = base_path / f"MetaTraderAccount{i}"
                terminal_path = account_dir / "terminal64.exe"
                data_path = account_dir / "MQL5" / "Data"
                
                print(f"\n📁 Setting up Account {i} for {symbol}...")
                
                # Create directory and copy files
                if not terminal_path.exists():
                    account_dir.mkdir(parents=True, exist_ok=True)
                    print(f"   📂 Copying MT5 installation...")
                    shutil.copytree(base_mt5_dir, account_dir, dirs_exist_ok=True)
                    print(f"   ✅ Copied to {account_dir}")
                else:
                    print(f"   ✅ Already exists: {account_dir}")
                
                # Update .env
                set_key(self.env_file, f'MT5_PATH_{symbol}', str(terminal_path))
                set_key(self.env_file, f'MT5_DATA_PATH_{symbol}', str(data_path))
                
                print(f"   ✅ Environment updated for {symbol}")
                success_count += 1
                
            except Exception as e:
                print(f"   ❌ Error setting up Account {i}: {e}")
        
        print(f"\n📊 Setup Results: {success_count}/{len(self.symbols)} accounts created")
        return success_count > 0
    
    def validate_setup(self):
        """Validate that all MT5 paths are properly configured"""
        print("🔍 VALIDATING MT5 PATH SETUP")
        print("="*35)
        
        valid_count = 0
        
        for symbol in self.symbols:
            path = os.getenv(f'MT5_PATH_{symbol}')
            
            if not path:
                print(f"❌ {symbol}: No path configured")
                continue
            
            # Handle launch commands with parameters
            if '/config:' in path:
                exe_path = path.split(' /config:')[0].strip('"')
            else:
                exe_path = path
            
            if os.path.exists(exe_path):
                print(f"✅ {symbol}: Path valid - {exe_path}")
                valid_count += 1
            else:
                print(f"❌ {symbol}: Path not found - {exe_path}")
        
        print(f"\n📊 Validation Results: {valid_count}/{len(self.symbols)} paths valid")
        return valid_count == len(self.symbols)

def main():
    """Main setup interface"""
    manager = MT5InstallationManager()
    
    while True:
        print("\n" + "="*60)
        print("🛠️ MT5 INSTALLATION PATH MANAGER")
        print("="*60)
        print("1. Show current path configuration")
        print("2. Create portable installations (recommended)")
        print("3. Create config-based setup (advanced)")
        print("4. Create simple directory setup")
        print("5. Validate current setup")
        print("6. Exit")
        print()
        
        choice = input("Select option (1-6): ").strip()
        
        if choice == '1':
            manager.show_current_paths()
        
        elif choice == '2':
            base_dir = input("Enter base directory (or Enter for D:/MT5_Portable): ").strip()
            if not base_dir:
                base_dir = "D:/MT5_Portable"
            manager.create_portable_installations(base_dir)
        
        elif choice == '3':
            base_dir = input("Enter config base directory (or Enter for D:/MT5_Configs): ").strip()
            if not base_dir:
                base_dir = "D:/MT5_Configs"
            manager.create_config_based_setup(base_dir)
        
        elif choice == '4':
            base_dir = input("Enter base directory (or Enter for D:/MetaTrader_Accounts): ").strip()
            if not base_dir:
                base_dir = "D:/MetaTrader_Accounts"
            manager.create_simple_directory_setup(base_dir)
        
        elif choice == '5':
            manager.validate_setup()
        
        elif choice == '6':
            print("👋 Setup complete!")
            break
        
        else:
            print("❌ Invalid choice, please try again")

if __name__ == "__main__":
    main()
