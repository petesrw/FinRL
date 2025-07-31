"""
Multi-Symbol Trading Launcher with Separate Accounts
Launches trading bots for different symbols using different MT5 accounts
"""

import os
import sys
import time
import threading
from datetime import datetime
from multi_account_mt5 import multi_mt5
from mt5_trading_bot import TradingBot
import signal

class MultiSymbolTradingLauncher:
    """
    Launches and manages multiple trading bots for different symbols
    Each symbol uses its own MT5 account
    """
    
    def __init__(self):
        self.trading_bots = {}  # symbol -> TradingBot instance
        self.bot_threads = {}   # symbol -> threading.Thread
        self.running = False
        self.symbol_configs = {
            'EURUSD': {'risk_percent': 0.3, 'max_positions': 2},
            'GBPUSD': {'risk_percent': 0.35, 'max_positions': 2}, 
            'USDJPY': {'risk_percent': 0.25, 'max_positions': 1},
            'XAUUSD': {'risk_percent': 0.4, 'max_positions': 3}
        }
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print(f"\n🛑 Received signal {signum}, shutting down...")
        self.stop_all_trading()
        sys.exit(0)
    
    def initialize_connections(self):
        """Initialize all MT5 connections"""
        print("🏦 INITIALIZING MULTI-ACCOUNT MT5 CONNECTIONS")
        print("="*50)
        
        # Test all connections
        success_count = 0
        for symbol in self.symbol_configs.keys():
            success, config = multi_mt5.get_connection_for_symbol(symbol)
            if success:
                success_count += 1
                print(f"✅ {symbol}: Connected to account {config['login']}")
            else:
                print(f"❌ {symbol}: Connection failed")
        
        if success_count == 0:
            print("❌ No MT5 connections successful!")
            return False
        
        print(f"\n✅ {success_count}/{len(self.symbol_configs)} accounts connected")
        multi_mt5.print_connection_summary()
        return True
    
    def create_trading_bot(self, symbol: str):
        """Create a trading bot for specific symbol with its own MT5 connection"""
        try:
            print(f"\n🤖 Creating trading bot for {symbol}...")
            
            # Verify connection for this symbol
            success, config = multi_mt5.get_connection_for_symbol(symbol)
            if not success:
                print(f"❌ Cannot create bot for {symbol}: No MT5 connection")
                return None
            
            # Get symbol-specific configuration
            symbol_config = self.symbol_configs.get(symbol, {})
            risk_percent = symbol_config.get('risk_percent', 0.3)
            
            # Create modified TradingBot that uses multi-account MT5
            bot = SymbolSpecificTradingBot(
                symbol=symbol,
                risk_percent=risk_percent,
                multi_mt5=multi_mt5
            )
            
            print(f"✅ Trading bot created for {symbol}")
            print(f"   Risk: {risk_percent}%")
            print(f"   Max Positions: {symbol_config.get('max_positions', 2)}")
            print(f"   Account: {config['login']}")
            
            return bot
            
        except Exception as e:
            print(f"❌ Error creating bot for {symbol}: {e}")
            return None
    
    def start_symbol_trading(self, symbol: str):
        """Start trading for specific symbol in separate thread"""
        bot = self.create_trading_bot(symbol)
        if bot is None:
            return False
        
        self.trading_bots[symbol] = bot
        
        # Create and start thread
        thread = threading.Thread(
            target=self._run_bot_loop,
            args=(symbol, bot),
            name=f"TradingBot-{symbol}",
            daemon=True
        )
        
        self.bot_threads[symbol] = thread
        thread.start()
        
        print(f"🚀 Started trading thread for {symbol}")
        return True
    
    def _run_bot_loop(self, symbol: str, bot):
        """Run trading loop for specific symbol"""
        try:
            print(f"🔄 Starting trading loop for {symbol}")
            
            while self.running:
                try:
                    # Run one trading iteration
                    result = bot.run_single_iteration()
                    
                    if result is None:
                        print(f"⚠️ {symbol}: Iteration returned None, retrying...")
                        time.sleep(30)
                        continue
                    
                    # Log result
                    if result.get('result', False):
                        action_names = {0: 'HOLD', 1: 'BUY', 2: 'SELL', 3: 'CLOSE'}
                        action_name = action_names.get(result['action'], 'UNKNOWN')
                        confidence = result.get('confidence', 0)
                        price = result.get('current_price', 0)
                        
                        print(f"📊 {symbol} @ {price:.5f}: {action_name} (conf: {confidence:.2f})")
                    
                    # Wait before next iteration (symbol-specific intervals)
                    if symbol == 'XAUUSD':
                        time.sleep(180)  # 3 minutes for Gold (more volatile)
                    elif symbol in ['EURUSD', 'GBPUSD']:
                        time.sleep(300)  # 5 minutes for major pairs
                    else:
                        time.sleep(360)  # 6 minutes for others
                        
                except Exception as e:
                    print(f"❌ Error in {symbol} trading loop: {e}")
                    time.sleep(60)  # Wait 1 minute before retrying
                    
        except Exception as e:
            print(f"❌ Fatal error in {symbol} trading thread: {e}")
        finally:
            print(f"🛑 Trading loop ended for {symbol}")
    
    def start_all_trading(self):
        """Start trading for all configured symbols"""
        if not self.initialize_connections():
            return False
        
        print("\n🚀 STARTING MULTI-SYMBOL TRADING")
        print("="*40)
        
        self.running = True
        
        # Start trading for each symbol
        for symbol in self.symbol_configs.keys():
            success = self.start_symbol_trading(symbol)
            if success:
                print(f"✅ {symbol} trading started")
                time.sleep(2)  # Small delay between starts
            else:
                print(f"❌ Failed to start {symbol} trading")
        
        print(f"\n🎯 {len(self.trading_bots)} trading bots active")
        return len(self.trading_bots) > 0
    
    def stop_symbol_trading(self, symbol: str):
        """Stop trading for specific symbol"""
        if symbol in self.trading_bots:
            bot = self.trading_bots[symbol]
            bot.running = False
            print(f"🛑 Stopping {symbol} trading...")
            
            # Wait for thread to finish
            if symbol in self.bot_threads:
                thread = self.bot_threads[symbol]
                thread.join(timeout=10)
                if thread.is_alive():
                    print(f"⚠️ {symbol} thread didn't stop gracefully")
                
                del self.bot_threads[symbol]
            
            del self.trading_bots[symbol]
            print(f"✅ {symbol} trading stopped")
    
    def stop_all_trading(self):
        """Stop all trading activities"""
        print("\n🛑 STOPPING ALL TRADING ACTIVITIES")
        print("="*40)
        
        self.running = False
        
        # Stop all bots
        symbols_to_stop = list(self.trading_bots.keys())
        for symbol in symbols_to_stop:
            self.stop_symbol_trading(symbol)
        
        # Disconnect all MT5 connections
        multi_mt5.disconnect_all()
        
        print("✅ All trading stopped")
    
    def get_trading_status(self):
        """Get status of all trading activities"""
        status = {
            'running': self.running,
            'active_symbols': list(self.trading_bots.keys()),
            'thread_status': {},
            'connection_status': multi_mt5.get_connection_status()
        }
        
        for symbol, thread in self.bot_threads.items():
            status['thread_status'][symbol] = {
                'alive': thread.is_alive(),
                'name': thread.name
            }
        
        return status
    
    def print_status_report(self):
        """Print comprehensive status report"""
        status = self.get_trading_status()
        
        print("\n" + "="*60)
        print("📊 MULTI-SYMBOL TRADING STATUS REPORT")
        print("="*60)
        print(f"System Running: {'✅ YES' if status['running'] else '❌ NO'}")
        print(f"Active Symbols: {len(status['active_symbols'])}")
        
        if status['active_symbols']:
            print("\n🤖 TRADING BOTS STATUS:")
            for symbol in status['active_symbols']:
                thread_alive = status['thread_status'].get(symbol, {}).get('alive', False)
                conn_status = status['connection_status'].get(symbol, {})
                connected = conn_status.get('connected', False)
                
                print(f"  {symbol}:")
                print(f"    Thread: {'✅ Active' if thread_alive else '❌ Stopped'}")
                print(f"    Connection: {'✅ Connected' if connected else '❌ Disconnected'}")
                print(f"    Account: {conn_status.get('account', 'Unknown')}")
        
        print("\n🏦 MT5 CONNECTIONS:")
        for symbol, conn_info in status['connection_status'].items():
            status_icon = "✅" if conn_info['connected'] else "❌"
            print(f"  {status_icon} {symbol}: Account {conn_info['account']} on {conn_info['server']}")

class SymbolSpecificTradingBot(TradingBot):
    """
    Modified TradingBot that uses multi-account MT5 system
    """
    
    def __init__(self, symbol, risk_percent=None, multi_mt5=None):
        self.multi_mt5 = multi_mt5
        # Initialize with symbol but don't connect MT5 yet
        self.symbol = symbol
        
        # Use multi-account MT5 instead of regular MT5Interface
        success, config = self.multi_mt5.get_connection_for_symbol(symbol)
        if not success:
            raise Exception(f"Failed to get MT5 connection for {symbol}")
        
        # Continue with regular TradingBot initialization but skip MT5 connection
        # (we'll override the MT5 methods to use multi_mt5)
        super().__init__(symbol, risk_percent=risk_percent)
    
    def _connect_mt5(self):
        """Override to use multi-account system"""
        success, config = self.multi_mt5.get_connection_for_symbol(self.symbol)
        return success

def main():
    """Main function to run multi-symbol trading"""
    launcher = MultiSymbolTradingLauncher()
    
    try:
        print("🏦 MULTI-SYMBOL FOREX TRADING SYSTEM")
        print("="*50)
        print("Each symbol uses its own MT5 account for better risk isolation")
        print()
        
        if launcher.start_all_trading():
            print("\n✅ Multi-symbol trading started successfully!")
            print("Press Ctrl+C to stop all trading")
            
            # Main monitoring loop
            while launcher.running:
                time.sleep(60)  # Status update every minute
                launcher.print_status_report()
                
        else:
            print("❌ Failed to start trading system")
            
    except KeyboardInterrupt:
        print("\n🛑 Shutdown requested by user")
    except Exception as e:
        print(f"❌ System error: {e}")
    finally:
        launcher.stop_all_trading()
        print("👋 Multi-symbol trading system shutdown complete")

if __name__ == "__main__":
    main()
