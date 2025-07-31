"""
Multi-Account MT5 Interface
Manages multiple MT5 accounts for different trading symbols
"""

import os
import time
import MetaTrader5 as mt5
from dotenv import load_dotenv
from typing import Dict, Optional, Tuple
import threading
from datetime import datetime

class MultiAccountMT5:
    """
    Manages multiple MT5 connections for different symbols
    Each symbol can use a different MT5 account
    """
    
    def __init__(self):
        load_dotenv()
        self.connections = {}  # symbol -> MT5 connection info
        self.active_connections = {}  # symbol -> connection object
        self.connection_locks = {}  # symbol -> threading lock
        self.symbol_account_mapping = {}
        
        # Load account configurations
        self._load_account_configs()
        
        # Initialize connection locks
        for symbol in self.symbol_account_mapping.keys():
            self.connection_locks[symbol] = threading.Lock()
    
    def _load_account_configs(self):
        """Load MT5 account configurations from environment"""
        
        # Load symbol-specific accounts
        symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD']
        
        for symbol in symbols:
            login_key = f'MT5_LOGIN_{symbol}'
            password_key = f'MT5_PASSWORD_{symbol}'
            server_key = f'MT5_SERVER_{symbol}'
            path_key = f'MT5_PATH_{symbol}'
            data_path_key = f'MT5_DATA_PATH_{symbol}'
            config_path_key = f'MT5_CONFIG_PATH_{symbol}'
            
            login = os.getenv(login_key)
            password = os.getenv(password_key)
            server = os.getenv(server_key)
            mt5_path = os.getenv(path_key)
            data_path = os.getenv(data_path_key)
            config_path = os.getenv(config_path_key)
            
            if login and password and server:
                try:
                    self.connections[symbol] = {
                        'login': int(login),
                        'password': password,
                        'server': server,
                        'mt5_path': mt5_path,
                        'data_path': data_path,
                        'config_path': config_path,
                        'connected': False,
                        'last_used': None,
                        'mt5_instance': None  # Store separate MT5 instance
                    }
                    self.symbol_account_mapping[symbol] = symbol
                    print(f"✅ Loaded config for {symbol}:")
                    print(f"   Account: {login} on {server}")
                    print(f"   MT5 Path: {mt5_path or 'Default'}")
                    print(f"   Data Path: {data_path or 'Default'}")
                except ValueError:
                    print(f"❌ Invalid login format for {symbol}: {login}")
            else:
                print(f"⚠️ Missing credentials for {symbol}")
        
        # Fallback to main account if symbol-specific not found
        main_login = os.getenv('MT5_LOGIN')
        main_password = os.getenv('MT5_PASSWORD')
        main_server = os.getenv('MT5_SERVER')
        main_path = os.getenv('MT5_PATH_WINDOWS_DEFAULT')
        
        if main_login and main_password and main_server:
            try:
                main_config = {
                    'login': int(main_login),
                    'password': main_password,
                    'server': main_server,
                    'mt5_path': main_path,
                    'data_path': None,
                    'config_path': None,
                    'connected': False,
                    'last_used': None,
                    'mt5_instance': None
                }
                
                # Use main account for any symbols without specific configs
                for symbol in ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD']:
                    if symbol not in self.connections:
                        self.connections[symbol] = main_config.copy()
                        self.symbol_account_mapping[symbol] = 'MAIN'
                        print(f"📋 Using main account for {symbol}: Account {main_login}")
                        
            except ValueError:
                print(f"❌ Invalid main login format: {main_login}")
    
    def get_connection_for_symbol(self, symbol: str) -> Tuple[bool, Optional[Dict]]:
        """
        Get or create MT5 connection for specific symbol
        Returns: (success, connection_info)
        """
        # Normalize symbol (remove suffixes like 'm', '.c')
        base_symbol = self._normalize_symbol(symbol)
        
        with self.connection_locks.get(base_symbol, threading.Lock()):
            # Check if we have a connection config for this symbol
            if base_symbol not in self.connections:
                print(f"❌ No MT5 configuration found for symbol {base_symbol}")
                return False, None
            
            config = self.connections[base_symbol]
            
            # Try to connect if not already connected
            if not config.get('connected', False):
                success = self._connect_to_account(base_symbol, config)
                if not success:
                    return False, None
            
            # Update last used time
            config['last_used'] = datetime.now()
            
            return True, config
    
    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol name for account mapping"""
        # Remove common suffixes
        symbol = symbol.replace('m', '').replace('.c', '').replace('.', '')
        return symbol.upper()
    
    def _connect_to_account(self, symbol: str, config: Dict) -> bool:
        """Connect to specific MT5 account with its own installation path and retry logic"""
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    print(f"� Retry attempt {attempt + 1}/{max_retries} for {symbol}")
                    time.sleep(retry_delay)
                
                print(f"�🔗 Connecting to MT5 for {symbol}...")
                print(f"   Account: {config['login']}")
                print(f"   Server: {config['server']}")
                print(f"   MT5 Path: {config.get('mt5_path', 'Default')}")
                
                # Initialize MT5 with specific path if provided
                mt5_path = config.get('mt5_path')
                if mt5_path and os.path.exists(mt5_path):
                    print(f"   🚀 Using custom MT5 installation: {mt5_path}")
                    # Shutdown previous connection first
                    mt5.shutdown()
                    time.sleep(1)
                    
                    # Initialize MT5 with specific path
                    if not mt5.initialize(path=mt5_path):
                        print(f"❌ Failed to initialize MT5 with path {mt5_path}")
                        # Try default initialization
                        if not mt5.initialize():
                            print(f"❌ Failed to initialize MT5 (default) for {symbol}")
                            if attempt < max_retries - 1:
                                continue
                            return False
                        else:
                            print(f"⚠️ Using default MT5 installation for {symbol}")
                    else:
                        print(f"✅ Initialized MT5 with custom path for {symbol}")
                else:
                    # Initialize MT5 if not already done or using default
                    if not mt5.initialize():
                        print(f"❌ Failed to initialize MT5 for {symbol}")
                        if attempt < max_retries - 1:
                            continue
                        return False
                    else:
                        print(f"✅ Initialized default MT5 for {symbol}")
                
                # Connect to the specific account
                if not mt5.login(config['login'], config['password'], config['server']):
                    error = mt5.last_error()
                    print(f"❌ Failed to login to MT5 for {symbol}: {error}")
                    
                    # Specific handling for common errors
                    if error[0] == -10005:  # IPC timeout
                        print(f"   💡 IPC timeout - server may be busy, retrying...")
                        if attempt < max_retries - 1:
                            continue
                    elif error[0] == 10004:  # No connection with trade server
                        print(f"   💡 No connection with trade server - retrying...")
                        if attempt < max_retries - 1:
                            continue
                    elif error[0] == 10003:  # Invalid parameters
                        print(f"   💡 Invalid parameters - check account credentials")
                        return False  # Don't retry for invalid credentials
                    
                    if attempt < max_retries - 1:
                        continue
                    return False
                
                # Verify connection
                account_info = mt5.account_info()
                if account_info is None:
                    print(f"❌ Failed to get account info for {symbol}")
                    if attempt < max_retries - 1:
                        continue
                    return False
                
                config['connected'] = True
                config['account_info'] = account_info
                
                print(f"✅ Connected to MT5 for {symbol}")
                print(f"   Account: {account_info.login}")
                print(f"   Balance: ${account_info.balance:.2f}")
                print(f"   Equity: ${account_info.equity:.2f}")
                print(f"   Server: {account_info.server}")
                print(f"   Company: {account_info.company}")
                
                return True
                
            except Exception as e:
                print(f"❌ Error connecting to MT5 for {symbol} (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    continue
        
        print(f"❌ Failed to connect to {symbol} after {max_retries} attempts")
        return False
    
    def get_symbol_rates(self, symbol: str, timeframe: int, count: int = 1000):
        """Get rates for symbol using appropriate account"""
        success, config = self.get_connection_for_symbol(symbol)
        if not success:
            return None
        
        try:
            # Get actual symbol name with suffix if needed
            actual_symbol = self._find_actual_symbol(symbol)
            if actual_symbol is None:
                return None
            
            rates = mt5.copy_rates_from_pos(actual_symbol, timeframe, 0, count)
            return rates
            
        except Exception as e:
            print(f"❌ Error getting rates for {symbol}: {e}")
            return None
    
    def _find_actual_symbol(self, symbol: str) -> Optional[str]:
        """Find the actual symbol name in MT5 (with correct suffix)"""
        # Try different variations - เพิ่มความหลากหลายมากขึ้น
        variations = [
            symbol,                           # ต้นฉบับ เช่น EURUSD
            symbol + 'm',                     # เช่น EURUSDm
            symbol + '.c',                    # เช่น EURUSD.c  
            symbol + '.e',                    # เช่น EURUSD.e
            symbol + '.raw',                  # เช่น EURUSD.raw
            symbol + '#',                     # เช่น EURUSD#
            symbol.replace('USD', 'USDm'),    # เช่น EURUSDm
            symbol.replace('USD', 'USD.c'),   # เช่น EURUSD.c
            symbol.replace('USD', 'USD.e'),   # เช่น EURUSD.e
            symbol.lower(),                   # เช่น eurusd
            symbol.lower() + 'm',             # เช่น eurusdm
            symbol.lower() + '.c',            # เช่น eurusd.c
        ]
        
        # สำหรับ Gold/Silver
        if symbol.upper() == 'XAUUSD':
            variations.extend(['GOLD', 'GOLD#', 'GOLD.c', 'GOLD.e', 'GOLDm', 'XAU_USD', 'XAU/USD'])
        elif symbol.upper() == 'XAGUSD':
            variations.extend(['SILVER', 'SILVER#', 'SILVER.c', 'SILVER.e', 'SILVERm', 'XAG_USD', 'XAG/USD'])
        
        # ลองหาใน MT5 symbols
        print(f"🔍 Searching for {symbol} variations...")
        for i, variation in enumerate(variations):
            try:
                symbol_info = mt5.symbol_info(variation)
                if symbol_info is not None:
                    print(f"✅ Found {symbol} as '{variation}' (variation #{i+1})")
                    # เลือก symbol ใน market watch
                    if not mt5.symbol_select(variation, True):
                        print(f"⚠️ Could not select {variation} in market watch")
                    return variation
                else:
                    print(f"   ❌ '{variation}' not found", end='\r')
            except Exception as e:
                print(f"   ❌ Error checking '{variation}': {e}", end='\r')
        
        print(f"\n⚠️ Symbol {symbol} not found in any of {len(variations)} variations")
        
        # ลองดู symbols ที่มีอยู่ใน MT5
        try:
            symbols = mt5.symbols_get()
            if symbols:
                matching_symbols = [s.name for s in symbols if symbol.upper() in s.name.upper()]
                if matching_symbols:
                    print(f"💡 Similar symbols found: {matching_symbols[:5]}")
                    return matching_symbols[0]  # ใช้ตัวแรกที่เจอ
        except Exception as e:
            print(f"❌ Error getting MT5 symbols: {e}")
        
        return None
    
    def send_order_for_symbol(self, symbol: str, order_type: int, volume: float, 
                            sl: float = None, tp: float = None, comment: str = ""):
        """Send order for specific symbol using appropriate account"""
        success, config = self.get_connection_for_symbol(symbol)
        if not success:
            return None
        
        try:
            actual_symbol = self._find_actual_symbol(symbol)
            if actual_symbol is None:
                return None
            
            # Get current price
            tick = mt5.symbol_info_tick(actual_symbol)
            if tick is None:
                print(f"❌ Failed to get tick for {actual_symbol}")
                return None
            
            # Determine price based on order type
            if order_type == mt5.ORDER_TYPE_BUY:
                price = tick.ask
            else:
                price = tick.bid
            
            # Create order request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": actual_symbol,
                "volume": volume,
                "type": order_type,
                "price": price,
                "deviation": 20,
                "magic": 234000 + hash(symbol) % 1000,  # Unique magic for each symbol
                "comment": comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Add SL and TP if provided
            if sl is not None:
                request["sl"] = sl
            if tp is not None:
                request["tp"] = tp
            
            # Send order
            result = mt5.order_send(request)
            
            if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"✅ Order sent for {symbol}: {result.order}")
                return result
            else:
                print(f"❌ Order failed for {symbol}: {result.retcode if result else 'No result'}")
                return result
                
        except Exception as e:
            print(f"❌ Error sending order for {symbol}: {e}")
            return None
    
    def get_positions_for_symbol(self, symbol: str):
        """Get positions for specific symbol"""
        success, config = self.get_connection_for_symbol(symbol)
        if not success:
            return None
        
        try:
            actual_symbol = self._find_actual_symbol(symbol)
            if actual_symbol is None:
                return None
            
            positions = mt5.positions_get(symbol=actual_symbol)
            return positions
            
        except Exception as e:
            print(f"❌ Error getting positions for {symbol}: {e}")
            return None
    
    def close_position_for_symbol(self, symbol: str, ticket: int):
        """Close specific position"""
        success, config = self.get_connection_for_symbol(symbol)
        if not success:
            return None
        
        try:
            # Get position
            position = mt5.positions_get(ticket=ticket)
            if not position:
                print(f"❌ Position {ticket} not found")
                return None
            
            position = position[0]
            actual_symbol = position.symbol
            
            # Get current price
            tick = mt5.symbol_info_tick(actual_symbol)
            if tick is None:
                return None
            
            # Determine close price and order type
            if position.type == mt5.POSITION_TYPE_BUY:
                close_price = tick.bid
                close_type = mt5.ORDER_TYPE_SELL
            else:
                close_price = tick.ask
                close_type = mt5.ORDER_TYPE_BUY
            
            # Create close request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": actual_symbol,
                "volume": position.volume,
                "type": close_type,
                "position": ticket,
                "price": close_price,
                "deviation": 20,
                "magic": position.magic,
                "comment": f"Close {symbol}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Send close order
            result = mt5.order_send(request)
            
            if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"✅ Position {ticket} closed for {symbol}")
                return result
            else:
                print(f"❌ Failed to close position {ticket} for {symbol}")
                return result
                
        except Exception as e:
            print(f"❌ Error closing position for {symbol}: {e}")
            return None
    
    def get_account_info_for_symbol(self, symbol: str):
        """Get account info for symbol's account"""
        success, config = self.get_connection_for_symbol(symbol)
        if not success:
            return None
        
        return config.get('account_info', mt5.account_info())
    
    def disconnect_all(self):
        """Disconnect all MT5 connections"""
        for symbol, config in self.connections.items():
            if config.get('connected', False):
                print(f"🔌 Disconnecting {symbol}")
                config['connected'] = False
        
        mt5.shutdown()
        print("🔌 All MT5 connections closed")
    
    def get_connection_status(self) -> Dict:
        """Get status of all connections"""
        status = {}
        for symbol, config in self.connections.items():
            status[symbol] = {
                'connected': config.get('connected', False),
                'account': config['login'],
                'server': config['server'],
                'last_used': config.get('last_used'),
                'account_mapping': self.symbol_account_mapping.get(symbol, 'UNKNOWN')
            }
        return status
    
    def print_connection_summary(self):
        """Print summary of all connections"""
        print("\n" + "="*60)
        print("🏦 MULTI-ACCOUNT MT5 CONNECTION SUMMARY")
        print("="*60)
        
        status = self.get_connection_status()
        
        for symbol, info in status.items():
            status_icon = "✅" if info['connected'] else "❌"
            print(f"{status_icon} {symbol}")
            print(f"   Account: {info['account']}")
            print(f"   Server: {info['server']}")
            print(f"   Mapping: {info['account_mapping']}")
            if info['last_used']:
                print(f"   Last Used: {info['last_used'].strftime('%H:%M:%S')}")
            print()

# Global instance
multi_mt5 = MultiAccountMT5()

if __name__ == "__main__":
    # Test the multi-account system
    multi_mt5.print_connection_summary()
    
    # Test connections
    test_symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD']
    
    for symbol in test_symbols:
        print(f"\n🧪 Testing connection for {symbol}...")
        success, config = multi_mt5.get_connection_for_symbol(symbol)
        if success:
            account_info = multi_mt5.get_account_info_for_symbol(symbol)
            if account_info:
                print(f"✅ {symbol}: Account {account_info.login}, Balance: ${account_info.balance:.2f}")
        else:
            print(f"❌ {symbol}: Connection failed")
    
    multi_mt5.print_connection_summary()
