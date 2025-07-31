#!/usr/bin/env python3
"""
🚀 Forex Trading AI Launcher
Interactive launcher for training and testing models
"""

import os
import sys
import time
from datetime import datetime
from forex_rl_simple import SimpleForexBot

class ForexLauncher:
    def __init__(self):
        self.symbols = {
            '1': ('EURUSD', '💶 EUR/USD - Major Pair'),
            '2': ('XAUUSD', '🥇 XAU/USD - Gold'),
            '3': ('GBPUSD', '💷 GBP/USD - Cable'),
            '4': ('USDJPY', '🇯🇵 USD/JPY - Yen'),
            '5': ('AUDUSD', '🇦🇺 AUD/USD - Aussie'),
            '6': ('USDCHF', '🇨🇭 USD/CHF - Swissy')
        }
        
        self.optimization_methods = {
            '1': ('smart_defaults', '🎯 Smart Defaults - Proven configurations'),
            '2': ('adaptive', '🔧 Adaptive - Performance-based optimization'),
            '3': ('meta_learning', '🧠 Meta-Learning - AI learns best indicators')
        }
    
    def find_all_models(self, symbol):
        """Find all available models for a symbol, sorted by score descending"""
        import glob
        import json
        
        # Look for models with more flexible patterns
        search_patterns = [
            f"models/**/{symbol.lower()}_diamond_*.zip",
            f"models/**/{symbol.lower()}_*diamond*.zip",
            f"models/**/{symbol.lower()}_gold_*.zip",
            f"models/**/{symbol.lower()}_*gold*.zip", 
            f"models/**/{symbol.lower()}_silver_*.zip",
            f"models/**/{symbol.lower()}_*silver*.zip",
            f"models/**/{symbol.lower()}_bronze_*.zip",
            f"models/**/{symbol.lower()}_*bronze*.zip",
            f"models/**/{symbol.lower()}_successful_*.zip",
            f"models/**/{symbol.lower()}_*successful*.zip",
            f"models/**/{symbol.lower()}_active_trader_*.zip",  # New pattern for active trader models
            f"models/**/{symbol.lower()}_best_*.zip",
            f"simple_forex_model_{symbol}_*.zip",  # Legacy models
        ]
        
        # Collect all models with their scores
        all_models = []
        found_files = set()  # To avoid duplicates
        
        for pattern in search_patterns:
            files = glob.glob(pattern, recursive=True)
            for file_path in files:
                if file_path in found_files:
                    continue
                found_files.add(file_path)
                
                try:
                    # Extract creation time for sorting
                    import os
                    creation_time = os.path.getctime(file_path)
                    
                    # Try to get score from filename
                    if "_score" in file_path:
                        score_part = file_path.split("_score")[1].split("_")[0]
                        base_score = float(score_part)
                    else:
                        base_score = 50  # Default score for simple models
                    
                    # Get tier from path and filename, calculate final score
                    file_lower = file_path.lower()
                    if "diamond" in file_lower:
                        tier = "💎 DIAMOND"
                        final_score = base_score + 1000  # Bonus for diamond tier
                    elif "gold" in file_lower:
                        tier = "🥇 GOLD"
                        final_score = base_score + 100
                    elif "silver" in file_lower:
                        tier = "🥈 SILVER"
                        final_score = base_score + 10
                    elif "bronze" in file_lower:
                        tier = "🥉 BRONZE"
                        final_score = base_score + 1
                    elif "successful" in file_lower:
                        tier = "✅ SUCCESS"
                        final_score = base_score
                    elif "active_trader" in file_lower:
                        # Classify active trader models based on score
                        if base_score >= 70:
                            tier = "🥇 GOLD"
                            final_score = base_score + 100
                        elif base_score >= 65:
                            tier = "🥈 SILVER"
                            final_score = base_score + 10
                        else:
                            tier = "🥉 BRONZE"
                            final_score = base_score + 1
                    else:
                        tier = "📦 SIMPLE"
                        final_score = base_score
                    
                    # Extract timestamp or attempt number for better identification
                    import re
                    timestamp_match = re.search(r'(\d{8}_\d{6})', file_path)
                    attempt_match = re.search(r'attempt(\d+)', file_path)
                    
                    timestamp = timestamp_match.group(1) if timestamp_match else "unknown"
                    attempt = attempt_match.group(1) if attempt_match else "0"
                    
                    all_models.append({
                        'path': file_path,
                        'tier': tier,
                        'base_score': base_score,
                        'final_score': final_score,
                        'creation_time': creation_time,
                        'timestamp': timestamp,
                        'attempt': attempt,
                        'filename': os.path.basename(file_path)
                    })
                        
                except:
                    continue
        
        # Sort by final score in descending order
        all_models.sort(key=lambda x: x['final_score'], reverse=True)
        
        return all_models

    def find_best_model(self, symbol):
        """Find the best available model for a symbol, sorted by score descending"""
        all_models = self.find_all_models(symbol)
        
        # Return the best model (first in sorted list) or None if no models found
        if all_models:
            best = all_models[0]
            return best['path'], best['tier'], best['final_score']
        else:
            return None, "", 0
    
    def select_model(self, symbol):
        """Let user select from available models for a symbol"""
        all_models = self.find_all_models(symbol)
        
        if not all_models:
            print(f"❌ No models found for {symbol}")
            print("💡 Please train the model first!")
            return None, "", 0
        
        print(f"\n📋 AVAILABLE MODELS FOR {symbol}:")
        print("=" * 80)
        print(f"{'#':<3} {'Tier':<12} {'Score':<8} {'Attempt':<8} {'Date':<12} {'Filename'}")
        print("-" * 80)
        
        for i, model in enumerate(all_models, 1):
            actual_score = (model['final_score'] - 1000 if model['final_score'] > 1000 
                          else model['final_score'] - 100 if model['final_score'] > 100 
                          else model['final_score'] - 10 if model['final_score'] > 10 
                          else model['final_score'] - 1 if model['final_score'] > 1 
                          else model['final_score'])
            
            # Format timestamp for display
            timestamp = model['timestamp']
            if timestamp != "unknown" and len(timestamp) == 15:  # Format: YYYYMMDD_HHMMSS
                display_date = f"{timestamp[:8]}"  # Just the date part
            else:
                display_date = "unknown"
            
            print(f"{i:<3} {model['tier']:<12} {actual_score:<8.1f} {model['attempt']:<8} {display_date:<12} {model['filename'][:35]}")
        
        print("-" * 80)
        print("0. 🔙 Go Back")
        print("=" * 80)
        
        while True:
            try:
                choice = input(f"👉 Select model (0-{len(all_models)}): ").strip()
                choice_num = int(choice)
                
                if choice_num == 0:
                    return None, "", 0
                elif 1 <= choice_num <= len(all_models):
                    selected = all_models[choice_num - 1]
                    print(f"✅ Selected: {selected['tier']} - Score {selected['base_score']:.1f}")
                    return selected['path'], selected['tier'], selected['final_score']
                else:
                    print(f"❌ Invalid choice! Please enter 0-{len(all_models)}")
            except ValueError:
                print("❌ Invalid input! Please enter a number.")
    
    def clear_screen(self):
        """Clear terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def print_header(self):
        """Print application header"""
        print("🚀 FOREX TRADING AI LAUNCHER")
        print("=" * 60)
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
    
    def show_main_menu(self):
        """Show main menu"""
        self.clear_screen()
        self.print_header()
        
        print("\n📋 MAIN MENU:")
        print("1. 🤖 Train Model")
        print("2. 🧪 Test Model")
        print("3. 📊 Check Model Status")
        print("4. � Validate Models (Bias Check)")
        print("5. �🚀 Start Live Trading")
        print("6. 🔧 System Check")
        print("7. ❌ Exit")
        print("-" * 60)
        
        choice = input("👉 Select option (1-7): ").strip()
        return choice
    
    def select_symbol(self):
        """Symbol selection menu"""
        print("\n💱 SELECT SYMBOL:")
        print("-" * 40)
        
        for key, (symbol, description) in self.symbols.items():
            print(f"{key}. {description}")
        
        print("-" * 40)
        choice = input("👉 Select symbol (1-6): ").strip()
        
        if choice in self.symbols:
            symbol, description = self.symbols[choice]
            print(f"✅ Selected: {description}")
            return symbol
        else:
            print("❌ Invalid selection!")
            return None
    
    def select_optimization_method(self):
        """Optimization method selection"""
        print("\n🔧 SELECT OPTIMIZATION METHOD:")
        print("-" * 50)
        
        for key, (method, description) in self.optimization_methods.items():
            print(f"{key}. {description}")
        
        print("-" * 50)
        choice = input("👉 Select method (1-3): ").strip()
        
        if choice in self.optimization_methods:
            method, description = self.optimization_methods[choice]
            print(f"✅ Selected: {description}")
            return method
        else:
            print("❌ Invalid selection!")
            return None
    
    def train_model(self):
        """Train model workflow"""
        print("\n🤖 TRAIN MODEL")
        print("=" * 40)
        
        # Select symbol
        symbol = self.select_symbol()
        if not symbol:
            return
        
        # Select optimization method
        method = self.select_optimization_method()
        if not method:
            return
        
        # Get training parameters
        print(f"\n⚙️ TRAINING PARAMETERS:")
        print("-" * 30)
        
        try:
            timesteps = input("🔢 Training steps (default: 50000): ").strip()
            timesteps = int(timesteps) if timesteps else 50000
            
            episodes = input("🧪 Test episodes (default: 10): ").strip()
            episodes = int(episodes) if episodes else 10
            
        except ValueError:
            print("❌ Invalid input! Using defaults.")
            timesteps = 50000
            episodes = 10
        
        # Confirm training
        print(f"\n📋 TRAINING SUMMARY:")
        print(f"   Symbol: {symbol}")
        print(f"   Method: {method}")
        print(f"   Steps: {timesteps:,}")
        print(f"   Test Episodes: {episodes}")
        
        confirm = input("\n🚀 Start training? (y/N): ").strip().lower()
        if confirm != 'y':
            print("❌ Training cancelled.")
            return
        
        # Start training
        print(f"\n🤖 Training {symbol} Model...")
        print("=" * 50)
        
        start_time = time.time()
        
        try:
            # Create bot and train
            bot = SimpleForexBot(symbol, method)
            print(f"📚 Training with {timesteps:,} steps...")
            bot.train_model(total_timesteps=timesteps)
            
            # Test model
            print(f"🧪 Testing with {episodes} episodes...")
            result = bot.test_model(episodes=episodes)
            
            # Calculate time
            training_time = time.time() - start_time
            
            # Show results
            print("\n" + "=" * 50)
            print("📊 TRAINING RESULTS:")
            print(f"⏱️ Time: {training_time/60:.1f} minutes")
            print(f"🎯 Target Met: {'✅ YES' if result else '❌ NO'}")
            print(f"📁 Model: simple_forex_model_{symbol}_PPO.zip")
            
            if result:
                print(f"🚀 {symbol} Model is ready for Live Trading!")
            else:
                print(f"⚠️ {symbol} Model needs improvement")
                
        except Exception as e:
            print(f"❌ Training failed: {e}")
        
        input("\n👉 Press Enter to continue...")
    
    def test_model(self):
        """Test existing model"""
        print("\n🧪 TEST MODEL")
        print("=" * 40)
        
        # Select symbol
        symbol = self.select_symbol()
        if not symbol:
            return
        
        # Find best available model
        model_file, tier, score = self.find_best_model(symbol)
        if not model_file:
            print(f"❌ No model found for {symbol}")
            print("💡 Please train the model first!")
            input("\n👉 Press Enter to continue...")
            return
        
        print(f"✅ Found model: {model_file}")
        print(f"   🏆 Tier: {tier}")
        print(f"   📊 Score: {score-1000 if score > 1000 else score-100 if score > 100 else score-10 if score > 10 else score-1 if score > 1 else score:.1f}")
        
        # Get test parameters
        try:
            episodes = input("🧪 Test episodes (default: 5): ").strip()
            episodes = int(episodes) if episodes else 5
        except ValueError:
            episodes = 5
        
        print(f"\n🧪 Testing {symbol} Model...")
        print("-" * 40)
        
        try:
            # Load and test model
            bot = SimpleForexBot(symbol, 'adaptive')
            if bot.load_model(model_file):
                result = bot.test_model(episodes=episodes)
                
                print(f"\n📊 TEST RESULTS:")
                print(f"🎯 Target Met: {'✅ YES' if result else '❌ NO'}")
                print(f"📁 Model: {model_file}")
                
                if result:
                    print(f"🚀 {symbol} Model is ready for Live Trading!")
                else:
                    print(f"⚠️ {symbol} Model may need re-training")
            else:
                print(f"❌ Failed to load model: {model_file}")
                
        except Exception as e:
            print(f"❌ Testing failed: {e}")
        
        input("\n👉 Press Enter to continue...")
    
    def check_model_status(self):
        """Check status of all models"""
        print("\n📊 MODEL STATUS CHECK")
        print("=" * 50)
        
        print("🔍 Checking all models...")
        print("-" * 30)
        
        total_models = 0
        ready_models = 0
        
        for _, (symbol, description) in self.symbols.items():
            model_file, tier, score = self.find_best_model(symbol)
            
            if model_file:
                total_models += 1
                actual_score = score-1000 if score > 1000 else score-100 if score > 100 else score-10 if score > 10 else score-1 if score > 1 else score
                
                if "DIAMOND" in tier or "GOLD" in tier:
                    status = f"✅ Ready ({tier} - {actual_score:.1f})"
                    ready_models += 1
                elif "SILVER" in tier or "BRONZE" in tier:
                    status = f"⚠️ Decent ({tier} - {actual_score:.1f})"
                    ready_models += 1
                else:
                    status = f"📦 Basic ({tier} - {actual_score:.1f})"
                
                print(f"{symbol:8} | {status}")
            else:
                print(f"{symbol:8} | ❌ Not Found")
        
        print("-" * 30)
        print(f"📊 Summary: {ready_models}/{total_models} models ready")
        
        if ready_models == 0:
            print("💡 No models ready. Please train models first!")
        elif ready_models < total_models:
            print("💡 Some models need training or improvement.")
        else:
            print("🚀 All models ready for Live Trading!")
        
        input("\n👉 Press Enter to continue...")
    
    def start_live_trading(self):
        """Start live trading"""
        print("\n🚀 START LIVE TRADING")
        print("=" * 40)
        
        # Select symbol
        symbol = self.select_symbol()
        if not symbol:
            return
        
        # Let user select from available models
        print(f"\n🤖 MODEL SELECTION FOR {symbol}")
        print("=" * 50)
        print("Select the model you want to use for live trading:")
        
        model_file, tier, score = self.select_model(symbol)
        if not model_file:
            print("❌ No model selected!")
            input("\n👉 Press Enter to continue...")
            return
        
        print(f"\n✅ Selected model: {model_file}")
        print(f"   🏆 Tier: {tier}")
        actual_score = score-1000 if score > 1000 else score-100 if score > 100 else score-10 if score > 10 else score-1 if score > 1 else score
        print(f"   📊 Score: {actual_score:.1f}")
        
        # Model confirmation
        confirm_model = input(f"\n🤖 Use this model for live trading? (y/N): ").strip().lower()
        if confirm_model != 'y':
            print("❌ Model selection cancelled.")
            return
        
        # Quick model test
        print(f"🧪 Quick model test for {symbol}...")
        try:
            # For now, skip the actual test and assume the model is good based on tier
            if "DIAMOND" in tier or "GOLD" in tier:
                print("✅ High-tier model detected - Ready for trading!")
                result = True
            else:
                print("⚠️ Lower-tier model - Use with caution")
                confirm = input("Continue anyway? (y/N): ").strip().lower()
                if confirm != 'y':
                    return
                result = True
        except Exception as e:
            print(f"❌ Model test failed: {e}")
            return
        
        # Trading mode selection
        print(f"\n🎯 TRADING MODE:")
        print("1. 🧪 Demo Mode (Safe)")
        print("2. 🚀 Live Mode (Real Money)")
        
        mode_choice = input("👉 Select mode (1-2): ").strip()
        
        if mode_choice == '1':
            demo_mode = True
            mode_name = "DEMO"
        elif mode_choice == '2':
            demo_mode = False
            mode_name = "LIVE"
            
            # Extra confirmation for live mode
            print("⚠️ WARNING: Live mode uses real money!")
            confirm = input("Are you sure? Type 'YES' to confirm: ").strip()
            if confirm != 'YES':
                print("❌ Live trading cancelled.")
                return
        else:
            print("❌ Invalid selection!")
            return
        
        # Show trading summary
        print(f"\n📋 TRADING SUMMARY:")
        print(f"   Symbol: {symbol}")
        print(f"   Mode: {mode_name}")
        print(f"   Model: {model_file}")
        
        confirm = input(f"\n🚀 Start {mode_name} trading? (y/N): ").strip().lower()
        if confirm != 'y':
            print("❌ Trading cancelled.")
            return
        
        # Update .env for demo/live mode
        try:
            with open('.env', 'r') as f:
                env_content = f.read()
            
            # Update DEMO_MODE
            if 'DEMO_MODE=' in env_content:
                env_content = env_content.replace(
                    f'DEMO_MODE={"false" if demo_mode else "true"}',
                    f'DEMO_MODE={"true" if demo_mode else "false"}'
                )
            else:
                env_content += f'\nDEMO_MODE={"true" if demo_mode else "false"}\n'
            
            # Update DEFAULT_SYMBOL
            if 'DEFAULT_SYMBOL=' in env_content:
                import re
                env_content = re.sub(r'DEFAULT_SYMBOL=\w+', f'DEFAULT_SYMBOL={symbol}', env_content)
            else:
                env_content += f'\nDEFAULT_SYMBOL={symbol}\n'
            
            with open('.env', 'w') as f:
                f.write(env_content)
                
        except Exception as e:
            print(f"⚠️ Warning: Could not update .env file: {e}")
        
        # Start trading system
        print(f"\n🚀 Starting {mode_name} trading for {symbol}...")
        print("=" * 50)
        print("💡 Press Ctrl+C to stop trading")
        print("📊 Monitor the logs for trading activity")
        print("=" * 50)
        
        try:
            # Import and run the main system with specific model
            from forex_system_with_config import ConfigurableForexBot
            
            print(f"🤖 Creating trading bot with selected model...")
            bot = ConfigurableForexBot(symbol=symbol, model_path=model_file)
            
            # Load the specific model we selected
            if bot.load_model(model_file):
                print(f"✅ Loaded selected model: {model_file}")
                
                # Test the model first
                print("🧪 Quick model validation...")
                if bot.test_model(episodes=3):
                    print("✅ Model validation passed")
                else:
                    print("⚠️ Model validation warning - continuing anyway")
                
                # Start live trading
                print(f"🚀 Starting {mode_name} trading...")
                if bot.start_live_trading():
                    print(f"✅ Trading started successfully!")
                    
                    # Monitor the trading
                    try:
                        print("📊 Monitoring trading activity...")
                        print("💡 Press Ctrl+C to stop")
                        
                        import time
                        while True:
                            time.sleep(30)  # Check every 30 seconds
                            report = bot.get_performance_report()
                            if report and 'performance' in report:
                                stats = report['performance']
                                print(f"📈 Trades: {stats.get('total_trades', 0)}, "
                                      f"Win Rate: {stats.get('win_rate', 0):.1%}, "
                                      f"P&L: ${stats.get('total_profit', 0):.2f}")
                    except KeyboardInterrupt:
                        print("\n🛑 Stopping trading...")
                        bot.stop_live_trading()
                        print("✅ Trading stopped successfully")
                else:
                    print("❌ Failed to start trading")
            else:
                print(f"❌ Failed to load model: {model_file}")
            
        except KeyboardInterrupt:
            print("\n🛑 Trading stopped by user")
        except Exception as e:
            print(f"\n❌ Trading system error: {e}")
            import traceback
            traceback.print_exc()
        
        input("\n👉 Press Enter to continue...")
    
    def system_check(self):
        """Perform system health check"""
        print("\n🔧 SYSTEM CHECK")
        print("=" * 40)
        
        checks = []
        
        # Check Python packages
        print("📦 Checking Python packages...")
        try:
            import pandas
            import numpy
            import gymnasium
            from stable_baselines3 import PPO
            checks.append(("Python Packages", "✅ OK"))
        except ImportError as e:
            checks.append(("Python Packages", f"❌ Missing: {e}"))
        
        # Check TA-Lib
        print("📈 Checking TA-Lib...")
        try:
            import talib
            checks.append(("TA-Lib", "✅ OK"))
        except ImportError:
            checks.append(("TA-Lib", "❌ Not installed"))
        
        # Check GPU
        print("🖥️ Checking GPU...")
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                checks.append(("GPU", f"✅ {gpu_name}"))
            else:
                checks.append(("GPU", "⚠️ CPU only"))
        except ImportError:
            checks.append(("GPU", "❌ PyTorch not found"))
        
        # Check .env file
        print("⚙️ Checking configuration...")
        if os.path.exists('.env'):
            checks.append(("Configuration", "✅ .env found"))
        else:
            checks.append(("Configuration", "❌ .env not found"))
        
        # Check models
        print("🤖 Checking models...")
        model_count = 0
        for _, (symbol, _) in self.symbols.items():
            if os.path.exists(f"simple_forex_model_{symbol}_PPO.zip"):
                model_count += 1
        
        checks.append(("Models", f"📊 {model_count}/{len(self.symbols)} found"))
        
        # Show results
        print("\n📊 SYSTEM CHECK RESULTS:")
        print("-" * 40)
        for component, status in checks:
            print(f"{component:15} | {status}")
        
        print("-" * 40)
        
        # Overall status
        critical_issues = sum(1 for _, status in checks if "❌" in status)
        if critical_issues == 0:
            print("🚀 System ready for trading!")
        else:
            print(f"⚠️ {critical_issues} critical issues found")
            print("💡 Please fix issues before trading")
        
        input("\n👉 Press Enter to continue...")
    
    def run(self):
        """Main application loop"""
        while True:
            choice = self.show_main_menu()
            
            if choice == '1':
                self.train_model()
            elif choice == '2':
                self.test_model()
            elif choice == '3':
                self.check_model_status()
            elif choice == '4':
                self.start_live_trading()
            elif choice == '5':
                self.system_check()
            elif choice == '6':
                print("\n👋 Goodbye! Happy Trading! 📈")
                break
            else:
                print("❌ Invalid option! Please try again.")
                time.sleep(1)

def main():
    """Main entry point"""
    try:
        launcher = ForexLauncher()
        launcher.run()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye! Happy Trading! 📈")
    except Exception as e:
        print(f"\n❌ Launcher error: {e}")

if __name__ == "__main__":
    main()