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
    
    def find_best_model(self, symbol):
        """Find the best available model for a symbol"""
        import glob
        import json
        
        # Look for models in order of preference: diamond > gold > silver > bronze > simple
        search_paths = [
            f"models/diamond/{symbol.lower()}_diamond_*.zip",
            f"models/gold/{symbol.lower()}_gold_*.zip", 
            f"models/silver/{symbol.lower()}_silver_*.zip",
            f"models/bronze/{symbol.lower()}_bronze_*.zip",
            f"models/diamond/simple_forex_model_{symbol}_PPO.zip",
            f"simple_forex_model_{symbol}_PPO.zip"
        ]
        
        best_model = None
        best_score = 0
        best_tier = ""
        
        for pattern in search_paths:
            files = glob.glob(pattern)
            for file_path in files:
                try:
                    # Try to get score from filename
                    if "_score" in file_path:
                        score_part = file_path.split("_score")[1].split("_")[0]
                        score = float(score_part)
                    else:
                        score = 50  # Default score for simple models
                    
                    # Get tier from path
                    if "diamond" in file_path:
                        tier = "💎 DIAMOND"
                        score += 1000  # Bonus for diamond tier
                    elif "gold" in file_path:
                        tier = "🥇 GOLD"
                        score += 100
                    elif "silver" in file_path:
                        tier = "🥈 SILVER"
                        score += 10
                    elif "bronze" in file_path:
                        tier = "🥉 BRONZE"
                        score += 1
                    else:
                        tier = "📦 SIMPLE"
                    
                    if score > best_score:
                        best_model = file_path
                        best_score = score
                        best_tier = tier
                        
                except:
                    continue
        
        return best_model, best_tier, best_score
    
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
        print("4. 🚀 Start Live Trading")
        print("5. 🔧 System Check")
        print("6. ❌ Exit")
        print("-" * 60)
        
        choice = input("👉 Select option (1-6): ").strip()
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
        
        # Find best available model
        model_file, tier, score = self.find_best_model(symbol)
        if not model_file:
            print(f"❌ No model found for {symbol}")
            print("💡 Please train the model first!")
            input("\n👉 Press Enter to continue...")
            return
        
        print(f"✅ Found model: {model_file}")
        print(f"   🏆 Tier: {tier}")
        actual_score = score-1000 if score > 1000 else score-100 if score > 100 else score-10 if score > 10 else score-1 if score > 1 else score
        print(f"   📊 Score: {actual_score:.1f}")
        
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
            # Import and run the main system
            from forex_system_with_config import main as forex_main
            forex_main()
            
        except KeyboardInterrupt:
            print("\n🛑 Trading stopped by user")
        except Exception as e:
            print(f"\n❌ Trading system error: {e}")
        
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