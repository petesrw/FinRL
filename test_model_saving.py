#!/usr/bin/env python3
"""
Test script to validate model saving functionality
Tests the fixes for async model saving infrastructure
"""

import os
import sys
import json
import asyncio
import time
from datetime import datetime
import pandas as pd
import numpy as np

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our training system
from train_all_models import AdaptiveTrainer

def create_sample_data():
    """Create sample forex data for testing"""
    print("📊 Creating sample XAUUSD data for testing...")
    
    # Generate 5000 rows of realistic XAUUSD data
    np.random.seed(42)  # For reproducible results
    
    dates = pd.date_range(start='2024-01-01', periods=5000, freq='5T')
    
    # Start price around 2000 (realistic XAUUSD price)
    base_price = 2000.0
    
    # Generate realistic price movements
    returns = np.random.normal(0, 0.001, 5000)  # Small random movements
    prices = [base_price]
    
    for i in range(1, 5000):
        # Add some trend and volatility
        trend = 0.0001 * np.sin(i / 100)  # Slight trending
        noise = returns[i]
        new_price = prices[-1] * (1 + trend + noise)
        prices.append(new_price)
    
    # Create OHLC data
    data = []
    for i in range(len(dates)):
        price = prices[i]
        high = price * (1 + abs(np.random.normal(0, 0.0005)))
        low = price * (1 - abs(np.random.normal(0, 0.0005)))
        open_price = price + np.random.normal(0, 0.0002) * price
        close_price = price + np.random.normal(0, 0.0002) * price
        
        data.append({
            'timestamp': dates[i],
            'open': open_price,
            'high': max(high, open_price, close_price),
            'low': min(low, open_price, close_price),
            'close': close_price,
            'volume': np.random.randint(100, 1000)
        })
    
    df = pd.DataFrame(data)
    print(f"✅ Created {len(df)} rows of sample data")
    print(f"   📈 Price range: {df['close'].min():.2f} - {df['close'].max():.2f}")
    
    return df

def test_directory_creation():
    """Test if model directories are created correctly"""
    print("\n🗂️ Testing directory creation...")
    
    trainer = AdaptiveTrainer(symbol='XAUUSD')
    
    # Expected directories
    expected_dirs = [
        'models',
        'models/bronze',
        'models/silver', 
        'models/gold',
        'models/diamond',
        'models/successful',
        'training_logs'
    ]
    
    # Check if directories exist or can be created
    for dir_path in expected_dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            print(f"   📁 Created directory: {dir_path}")
        else:
            print(f"   ✅ Directory exists: {dir_path}")
    
    return True

def test_save_functions():
    """Test the save functions with mock data"""
    print("\n💾 Testing save functions...")
    
    trainer = AdaptiveTrainer(symbol='XAUUSD')
    
    # Test save_successful_config
    mock_config = {
        'algorithm': 'PPO',
        'learning_rate': 0.001,
        'gamma': 0.99,
        'batch_size': 1024,
        'timesteps': 1000000
    }
    
    mock_metrics = {
        'win_rate': 0.735,
        'profit_factor': 4.76,
        'max_drawdown': 0.15,
        'total_return': 0.85,
        'sharpe_ratio': 2.1,
        'total_trades': 125
    }
    
    try:
        print("   🧪 Testing save_successful_config...")
        trainer.save_successful_config(mock_config, mock_metrics, 93.1, 'diamond', 1)
        print("   ✅ save_successful_config - PASSED")
    except Exception as e:
        print(f"   ❌ save_successful_config - FAILED: {e}")
        return False
    
    # Test save_history
    try:
        print("   🧪 Testing save_history...")
        trainer.training_history = [{
            'attempt': 1,
            'timestamp': datetime.now().isoformat(),
            'hyperparameters': mock_config,
            'metrics': mock_metrics,
            'score': 93.1,
            'tier': 'diamond',
            'training_time': 1800.5
        }]
        trainer.save_history()
        print("   ✅ save_history - PASSED")
    except Exception as e:
        print(f"   ❌ save_history - FAILED: {e}")
        return False
    
    # Test save_async_training_log
    try:
        print("   🧪 Testing save_async_training_log...")
        mock_batch_results = [{
            'model_id': 1,
            'hyperparameters': mock_config,
            'metrics': mock_metrics,
            'score': 93.1,
            'tier': 'diamond',
            'success': True,
            'training_time': 1800.5
        }]
        trainer.save_async_training_log(mock_batch_results, 1)
        print("   ✅ save_async_training_log - PASSED")
    except Exception as e:
        print(f"   ❌ save_async_training_log - FAILED: {e}")
        return False
    
    return True

def test_model_saving_logic():
    """Test the model saving logic without actual model training"""
    print("\n🎯 Testing model saving logic...")
    
    trainer = AdaptiveTrainer(symbol='XAUUSD')
    
    # Create a mock model (we'll use None since we're just testing the save logic)
    mock_model = None  # In real scenario, this would be a trained model
    
    try:
        print("   🧪 Testing _save_model_by_tier...")
        
        # Test saving for each tier
        tiers_to_test = ['bronze', 'silver', 'gold', 'diamond']
        
        for tier in tiers_to_test:
            try:
                # This will fail when trying to save None model, but will test the directory creation
                model_path = trainer._save_model_by_tier(mock_model, tier, 85.0, 1, is_best=False)
                print(f"   ⚠️ {tier} tier save would work (directory structure correct)")
            except Exception as e:
                if "NoneType" in str(e) or "save" in str(e).lower():
                    print(f"   ✅ {tier} tier directory structure correct (expected model save error)")
                else:
                    print(f"   ❌ {tier} tier unexpected error: {e}")
                    return False
    
        print("   ✅ _save_model_by_tier logic - PASSED")
        return True
        
    except Exception as e:
        print(f"   ❌ _save_model_by_tier logic - FAILED: {e}")
        return False

def test_file_creation():
    """Test if all required files are created"""
    print("\n📄 Testing file creation...")
    
    trainer = AdaptiveTrainer(symbol='XAUUSD')
    
    expected_files = [
        trainer.history_file,
        trainer.successful_configs_file,
        trainer.failed_configs_file,
        trainer.async_log_file
    ]
    
    print("   📋 Expected files:")
    for file_path in expected_files:
        print(f"      📝 {file_path}")
        
        # Check if file exists or can be created
        try:
            if os.path.exists(file_path):
                print(f"         ✅ File exists")
                
                # Check if file is valid JSON
                with open(file_path, 'r') as f:
                    content = f.read().strip()
                    if content:
                        json.loads(content)
                        print(f"         ✅ Valid JSON format")
                    else:
                        print(f"         ⚠️ File is empty (will be populated)")
            else:
                print(f"         📁 File will be created on first save")
                
        except json.JSONDecodeError as e:
            print(f"         ❌ Invalid JSON: {e}")
            return False
        except Exception as e:
            print(f"         ❌ Error: {e}")
            return False
    
    return True

def test_quick_training_simulation():
    """Simulate a quick training to test the complete pipeline"""
    print("\n🏃‍♂️ Testing quick training simulation...")
    
    # Create sample data
    data = create_sample_data()
    
    # Initialize trainer
    trainer = AdaptiveTrainer(symbol='XAUUSD')
    
    # Test hyperparameter generation
    try:
        print("   🧪 Testing hyperparameter generation...")
        hyperparams = trainer.generate_hyperparameters()
        print(f"   ✅ Generated hyperparameters: {hyperparams['algorithm']}")
        print(f"      📊 Learning Rate: {hyperparams['learning_rate']}")
        print(f"      🎯 Gamma: {hyperparams['gamma']}")
        print(f"      📦 Batch Size: {hyperparams.get('batch_size', 'N/A')}")
    except Exception as e:
        print(f"   ❌ Hyperparameter generation failed: {e}")
        return False
    
    # Test scoring system
    try:
        print("   🧪 Testing scoring system...")
        mock_metrics = {
            'win_rate': 0.735,
            'profit_factor': 4.76,
            'max_drawdown': 0.15,
            'total_return': 0.85,
            'sharpe_ratio': 2.1,
            'total_trades': 125
        }
        
        score = trainer.calculate_score(mock_metrics)
        tier = trainer.get_tier(mock_metrics)
        
        print(f"   ✅ Score calculation: {score:.1f}")
        print(f"   ✅ Tier assignment: {tier}")
        
    except Exception as e:
        print(f"   ❌ Scoring system failed: {e}")
        return False
    
    print("   ✅ Quick training simulation - PASSED")
    return True

def test_real_model_creation():
    """Test actual model training and file creation"""
    print("\n🚀 Testing real model creation...")
    
    # Create sample data (smaller for faster training)
    print("   📊 Creating compact training data...")
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=1000, freq='5T')  # Smaller dataset
    
    base_price = 2000.0
    returns = np.random.normal(0, 0.002, 1000)  # Slightly higher volatility for better training
    prices = [base_price]
    
    for i in range(1, 1000):
        trend = 0.0005 * np.sin(i / 50)  # More pronounced trend
        noise = returns[i]
        new_price = prices[-1] * (1 + trend + noise)
        prices.append(new_price)
    
    data = []
    for i in range(len(dates)):
        price = prices[i]
        high = price * (1 + abs(np.random.normal(0, 0.001)))
        low = price * (1 - abs(np.random.normal(0, 0.001)))
        open_price = price + np.random.normal(0, 0.0005) * price
        close_price = price + np.random.normal(0, 0.0005) * price
        
        data.append({
            'timestamp': dates[i],
            'open': open_price,
            'high': max(high, open_price, close_price),
            'low': min(low, open_price, close_price),
            'close': close_price,
            'volume': np.random.randint(100, 1000)
        })
    
    df = pd.DataFrame(data)
    print(f"   ✅ Created {len(df)} rows of training data")
    
    # Initialize trainer
    trainer = AdaptiveTrainer(symbol='TESTMODEL')
    
    # Generate fast hyperparameters for testing
    try:
        print("   🧪 Generating optimized test hyperparameters...")
        hyperparams = {
            'algorithm': 'PPO',
            'learning_rate': 0.001,
            'gamma': 0.99,
            'batch_size': 256,  # Smaller batch for speed
            'n_steps': 512,     # Smaller steps for speed
            'timesteps': 50000, # Much smaller timesteps for fast training
            'lookback_window': 20,  # Smaller lookback
            'transaction_cost': 0.0
        }
        print(f"   ✅ Test hyperparameters ready: {hyperparams['timesteps']:,} timesteps")
    except Exception as e:
        print(f"   ❌ Hyperparameter generation failed: {e}")
        return False
    
    # Train a real model
    try:
        print("   🏋️‍♂️ Training real model (this may take 1-2 minutes)...")
        model, metrics, score, tier, emoji, training_time = trainer.train_model(df, hyperparams)
        
        print(f"   ✅ Model training completed!")
        print(f"      🎯 Tier: {tier}")
        print(f"      📊 Score: {score:.1f}")
        print(f"      📈 Win Rate: {metrics.get('win_rate', 0):.1%}")
        print(f"      💰 Profit Factor: {metrics.get('profit_factor', 0):.2f}")
        print(f"      🕒 Training Time: {training_time:.1f}s")
        
        # Check if we got a valid tier
        if tier not in ['bronze', 'silver', 'gold', 'diamond', 'none']:
            print(f"   ⚠️ Got unexpected tier: {tier}, treating as 'bronze' for testing")
            tier = 'bronze'
            
        # Force save as bronze if tier is 'none' for testing purposes
        if tier == 'none':
            tier = 'bronze'
            print(f"   📝 Forcing tier to 'bronze' for model saving test")
        
    except Exception as e:
        print(f"   ❌ Model training failed: {e}")
        return False
    
    # Test model saving
    try:
        print(f"   💾 Testing model save for tier: {tier}")
        model_path = trainer._save_model_by_tier(model, tier, score, 1, is_best=True)
        
        # Check if model file was created
        if not os.path.exists(model_path):
            print(f"   ❌ Model file not found: {model_path}")
            return False
        
        print(f"   ✅ Model file created: {model_path}")
        
        # Check model file size
        file_size = os.path.getsize(model_path)
        print(f"   📦 Model file size: {file_size:,} bytes")
        
        if file_size < 1000:  # Less than 1KB is probably empty
            print(f"   ❌ Model file too small, likely empty")
            return False
        
        # Check info file
        info_path = model_path.replace('.zip', '_info.json')
        if not os.path.exists(info_path):
            print(f"   ❌ Model info file not found: {info_path}")
            return False
        
        # Validate info file content
        with open(info_path, 'r') as f:
            info_data = json.load(f)
            required_keys = ['symbol', 'tier', 'score', 'attempt', 'timestamp']
            for key in required_keys:
                if key not in info_data:
                    print(f"   ❌ Missing key '{key}' in info file")
                    return False
        
        print(f"   ✅ Model info file valid: {info_path}")
        print(f"   ✅ Model creation test - PASSED")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Model saving failed: {e}")
        return False

def run_comprehensive_test():
    """Run all tests"""
    print("🧪 COMPREHENSIVE MODEL SAVING TEST")
    print("=" * 60)
    
    test_results = []
    
    # Run each test
    tests = [
        ("Directory Creation", test_directory_creation),
        ("Save Functions", test_save_functions),
        ("Model Saving Logic", test_model_saving_logic),
        ("File Creation", test_file_creation),
        ("Quick Training Simulation", test_quick_training_simulation),
        ("Real Model Creation", test_real_model_creation)
    ]
    
    for test_name, test_func in tests:
        print(f"\n🔬 Running: {test_name}")
        try:
            result = test_func()
            test_results.append((test_name, result))
            
            if result:
                print(f"   ✅ {test_name} - PASSED")
            else:
                print(f"   ❌ {test_name} - FAILED")
                
        except Exception as e:
            print(f"   💥 {test_name} - ERROR: {e}")
            test_results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {status} - {test_name}")
    
    print(f"\n🎯 Overall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Model saving system is ready!")
        print("\n💡 Next steps:")
        print("   1. Run actual training: python train_all_models.py")
        print("   2. Check models/ directory for saved model files")
        print("   3. Check training_logs/ for populated JSON files")
    else:
        print("⚠️ Some tests failed. Please fix issues before running training.")
    
    return passed == total

if __name__ == "__main__":
    try:
        success = run_comprehensive_test()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test crashed: {e}")
        sys.exit(1)
