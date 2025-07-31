"""
MT5 Multi-Account Configuration Helper
Helps you set up different MT5 accounts for different symbols
"""

import os
from dotenv import load_dotenv, set_key

class MT5AccountConfigurator:
    """Helper to configure multiple MT5 accounts"""
    
    def __init__(self, env_file=".env"):
        self.env_file = env_file
        load_dotenv(env_file)
        
        self.symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD']
        self.current_configs = {}
        self._load_current_configs()
    
    def _load_current_configs(self):
        """Load current configurations from .env"""
        for symbol in self.symbols:
            login = os.getenv(f'MT5_LOGIN_{symbol}')
            password = os.getenv(f'MT5_PASSWORD_{symbol}')
            server = os.getenv(f'MT5_SERVER_{symbol}')
            
            if login and password and server:
                self.current_configs[symbol] = {
                    'login': login,
                    'password': password,
                    'server': server
                }
    
    def show_current_config(self):
        """Display current configuration"""
        print("📋 CURRENT MT5 ACCOUNT CONFIGURATION")
        print("="*50)
        
        for symbol in self.symbols:
            if symbol in self.current_configs:
                config = self.current_configs[symbol]
                print(f"✅ {symbol}:")
                print(f"   Login: {config['login']}")
                print(f"   Password: {config['password']}")
                print(f"   Server: {config['server']}")
            else:
                print(f"❌ {symbol}: Not configured")
            print()
    
    def configure_symbol_account(self, symbol: str, login: str, password: str, server: str, mt5_path: str = None):
        """Configure MT5 account for specific symbol including path"""
        try:
            # Validate login is numeric
            int(login)
            
            # Set environment variables
            set_key(self.env_file, f'MT5_LOGIN_{symbol}', login)
            set_key(self.env_file, f'MT5_PASSWORD_{symbol}', password)
            set_key(self.env_file, f'MT5_SERVER_{symbol}', server)
            
            # Set MT5 path if provided
            if mt5_path:
                set_key(self.env_file, f'MT5_PATH_{symbol}', mt5_path)
            
            # Update current configs
            self.current_configs[symbol] = {
                'login': login,
                'password': password,
                'server': server,
                'mt5_path': mt5_path
            }
            
            print(f"✅ {symbol} account configured:")
            print(f"   Login: {login}")
            print(f"   Server: {server}")
            print(f"   MT5 Path: {mt5_path or 'Default'}")
            
            return True
            
        except ValueError:
            print(f"❌ Invalid login format: {login} (must be numeric)")
            return False
        except Exception as e:
            print(f"❌ Error configuring {symbol}: {e}")
            return False
    
    def interactive_setup(self):
        """Interactive setup of all accounts"""
        print("🔧 INTERACTIVE MT5 ACCOUNT SETUP")
        print("="*40)
        print("Configure MT5 accounts for each trading symbol")
        print("Leave blank to skip a symbol or use existing configuration")
        print()
        
        for symbol in self.symbols:
            print(f"\n📊 Configuring {symbol}")
            print("-" * 20)
            
            # Show current config if exists
            if symbol in self.current_configs:
                current = self.current_configs[symbol]
                print(f"Current: {current['login']} on {current['server']}")
                if 'mt5_path' in current and current['mt5_path']:
                    print(f"MT5 Path: {current['mt5_path']}")
            
            # Get new configuration
            login = input(f"MT5 Login for {symbol} (or Enter to skip): ").strip()
            if not login:
                print(f"⏭️ Skipping {symbol}")
                continue
            
            password = input(f"MT5 Password for {symbol}: ").strip()
            if not password:
                print(f"❌ Password required for {symbol}")
                continue
            
            server = input(f"MT5 Server for {symbol} (e.g., Exness-MT5Trial14): ").strip()
            if not server:
                print(f"❌ Server required for {symbol}")
                continue
            
            mt5_path = input(f"MT5 Path for {symbol} (or Enter for default): ").strip()
            
            # Configure the account
            success = self.configure_symbol_account(symbol, login, password, server, mt5_path)
            if success:
                print(f"✅ {symbol} configured successfully!")
            else:
                print(f"❌ Failed to configure {symbol}")
    
    def quick_setup_template(self):
        """Create a template for quick setup"""
        print("📝 QUICK SETUP TEMPLATE")
        print("="*30)
        print("Copy this template and modify with your actual account details:")
        print()
        
        base_login = 272015671
        base_server = "Exness-MT5Trial"
        password = "P@ssw0rd"
        
        for i, symbol in enumerate(self.symbols):
            login = base_login + i
            server = f"{base_server}{14 + i}"
            
            print(f"# {symbol} Account")
            print(f"MT5_LOGIN_{symbol}={login}")
            print(f"MT5_PASSWORD_{symbol}={password}")
            print(f"MT5_SERVER_{symbol}={server}")
            print()
    
    def validate_all_configs(self):
        """Validate all account configurations"""
        print("🔍 VALIDATING ACCOUNT CONFIGURATIONS")
        print("="*40)
        
        valid_count = 0
        
        for symbol in self.symbols:
            if symbol in self.current_configs:
                config = self.current_configs[symbol]
                
                # Basic validation
                try:
                    int(config['login'])  # Must be numeric
                    
                    if len(config['password']) < 4:
                        print(f"⚠️ {symbol}: Password seems too short")
                    
                    if not config['server']:
                        print(f"❌ {symbol}: Server is empty")
                        continue
                    
                    print(f"✅ {symbol}: Configuration valid")
                    valid_count += 1
                    
                except ValueError:
                    print(f"❌ {symbol}: Invalid login format")
            else:
                print(f"❌ {symbol}: Not configured")
        
        print(f"\n📊 {valid_count}/{len(self.symbols)} accounts configured and valid")
        return valid_count == len(self.symbols)
    
    def copy_main_account_to_all(self):
        """Copy main account credentials to all symbols (for testing)"""
        main_login = os.getenv('MT5_LOGIN')
        main_password = os.getenv('MT5_PASSWORD')
        main_server = os.getenv('MT5_SERVER')
        
        if not all([main_login, main_password, main_server]):
            print("❌ Main MT5 account not configured in .env")
            return False
        
        print("📋 Copying main account to all symbols...")
        print(f"Main Account: {main_login} on {main_server}")
        
        confirm = input("Are you sure? This will overwrite existing configs (y/N): ")
        if confirm.lower() != 'y':
            print("❌ Operation cancelled")
            return False
        
        success_count = 0
        for symbol in self.symbols:
            if self.configure_symbol_account(symbol, main_login, main_password, main_server):
                success_count += 1
        
        print(f"✅ {success_count}/{len(self.symbols)} symbols configured with main account")
        return success_count == len(self.symbols)

def main():
    """Main configuration interface"""
    configurator = MT5AccountConfigurator()
    
    while True:
        print("\n" + "="*60)
        print("🏦 MT5 MULTI-ACCOUNT CONFIGURATION")
        print("="*60)
        print("1. Show current configuration")
        print("2. Interactive setup (configure each symbol)")
        print("3. Quick setup template")
        print("4. Copy main account to all symbols")
        print("5. Validate all configurations")
        print("6. Exit")
        print()
        
        choice = input("Select option (1-6): ").strip()
        
        if choice == '1':
            configurator.show_current_config()
        
        elif choice == '2':
            configurator.interactive_setup()
        
        elif choice == '3':
            configurator.quick_setup_template()
        
        elif choice == '4':
            configurator.copy_main_account_to_all()
        
        elif choice == '5':
            configurator.validate_all_configs()
        
        elif choice == '6':
            print("👋 Configuration complete!")
            break
        
        else:
            print("❌ Invalid choice, please try again")

if __name__ == "__main__":
    main()
