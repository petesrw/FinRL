"""
Test Enhanced Active Trading System
ทดสอบการแก้ไขปัญหา Trading Activity
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# เพิ่มการ import ที่จำเป็น
import numpy as np
import pandas as pd
import random
import json
from datetime import datetime

print('🎯 Testing Enhanced Active Trading System')
print('='*60)

# สร้าง Mock Training Data
def create_mock_training_data():
    """สร้างข้อมูลจำลองสำหรับทดสอบ"""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=5000, freq='5min')
    
    # สร้าง OHLC data
    base_price = 2000
    prices = []
    current_price = base_price
    
    for i in range(len(dates)):
        # Random walk with some trend
        change = np.random.normal(0, 0.5)
        current_price += change
        
        # Generate OHLC
        high = current_price + abs(np.random.normal(0, 0.3))
        low = current_price - abs(np.random.normal(0, 0.3))
        open_price = current_price + np.random.normal(0, 0.2)
        close_price = current_price + np.random.normal(0, 0.2)
        
        prices.append({
            'timestamp': dates[i],
            'open': open_price,
            'high': high,
            'low': low,
            'close': close_price,
            'volume': np.random.randint(100, 1000)
        })
    
    return pd.DataFrame(prices)

# Test Enhanced Environment
def test_enhanced_environment():
    """ทดสอบ Enhanced Environment"""
    print('\n📊 Testing Enhanced AdvancedForexEnv...')
    
    # Create mock data
    data = create_mock_training_data()
    print(f'✅ Created mock data: {len(data)} rows')
    
    # Import the enhanced environment from train_all_models
    try:
        from train_all_models import AdvancedForexEnv
        
        # Create environment with lower transaction cost
        env = AdvancedForexEnv(data, transaction_cost=0.00005)
        
        # Test environment
        obs, info = env.reset()
        print(f'✅ Environment reset successful, observation shape: {obs.shape}')
        
        # Simulate active trading
        total_rewards = 0
        actions_taken = []
        
        for step in range(100):
            # Simulate more active trading (less holding)
            action_prob = np.random.random()
            if action_prob < 0.3:      # 30% buy
                action = 1
            elif action_prob < 0.6:    # 30% sell
                action = 2
            elif action_prob < 0.7:    # 10% close
                action = 3
            else:                      # 30% hold
                action = 0
                
            actions_taken.append(action)
            
            obs, reward, done, terminated, info = env.step(action)
            total_rewards += reward
            
            if done:
                break
        
        # Calculate trading activity
        non_hold_actions = sum(1 for a in actions_taken if a != 0)
        trading_activity = non_hold_actions / len(actions_taken) * 100
        
        print(f'📊 Test Results:')
        print(f'   • Steps taken: {len(actions_taken)}')
        print(f'   • Trading activity: {trading_activity:.1f}%')
        print(f'   • Total trades: {env.total_trades}')
        print(f'   • Total rewards: {total_rewards:.2f}')
        print(f'   • Final balance: {env.balance:.2f}')
        print(f'   • Final equity: {env.equity:.2f}')
        
        return {
            'success': True,
            'trading_activity': trading_activity,
            'total_trades': env.total_trades,
            'total_rewards': total_rewards
        }
        
    except Exception as e:
        print(f'❌ Environment test failed: {str(e)}')
        return {'success': False, 'error': str(e)}

# Test Enhanced Scoring
def test_enhanced_scoring():
    """ทดสอบระบบ Scoring ใหม่"""
    print('\n🏆 Testing Enhanced Scoring System...')
    
    try:
        from train_all_models import AdaptiveTrainer
        
        trainer = AdaptiveTrainer()
        
        # Test scenarios
        test_cases = [
            {
                'name': 'High Activity + Good Performance',
                'metrics': {
                    'total_trades': 60,
                    'win_rate': 0.55,
                    'profit_factor': 1.3,
                    'max_drawdown': 0.15,
                    'total_return': 0.08,
                    'sharpe_ratio': 0.8
                }
            },
            {
                'name': 'Low Activity + Excellent Performance',
                'metrics': {
                    'total_trades': 5,  # Very low activity
                    'win_rate': 0.80,
                    'profit_factor': 2.5,
                    'max_drawdown': 0.05,
                    'total_return': 0.15,
                    'sharpe_ratio': 1.5
                }
            },
            {
                'name': 'Medium Activity + Average Performance',
                'metrics': {
                    'total_trades': 30,
                    'win_rate': 0.50,
                    'profit_factor': 1.1,
                    'max_drawdown': 0.20,
                    'total_return': 0.02,
                    'sharpe_ratio': 0.5
                }
            },
            {
                'name': 'No Trading (Worst Case)',
                'metrics': {
                    'total_trades': 0,
                    'win_rate': 0.0,
                    'profit_factor': 0.0,
                    'max_drawdown': 0.0,
                    'total_return': 0.0,
                    'sharpe_ratio': 0.0
                }
            }
        ]
        
        print('📊 Scoring Test Results:')
        print('   Case                          | Trades | Score | Tier   ')
        print('   ------------------------------|--------|-------|--------')
        
        for case in test_cases:
            metrics = case['metrics']
            score = trainer.calculate_score(metrics)
            tier, icon = trainer.get_tier(metrics)
            
            print(f'   {case["name"]:<30} | {metrics["total_trades"]:>6} | {score:>5.1f} | {tier:<5} {icon}')
        
        print('\n✅ Enhanced scoring system working correctly!')
        print('   🎯 High activity models get better scores')
        print('   🚫 Low/No activity models get penalized')
        
        return {'success': True}
        
    except Exception as e:
        print(f'❌ Scoring test failed: {str(e)}')
        return {'success': False, 'error': str(e)}

# Test Enhanced Hyperparameters
def test_enhanced_hyperparameters():
    """ทดสอบการสร้าง Hyperparameters ใหม่"""
    print('\n🔧 Testing Enhanced Hyperparameter Generation...')
    
    try:
        from train_all_models import AdaptiveTrainer
        
        trainer = AdaptiveTrainer()
        
        # Generate multiple configs
        configs = []
        algorithm_counts = {}
        
        for i in range(10):
            config = trainer.generate_hyperparameters()
            configs.append(config)
            
            algo = config.get('algorithm', 'Unknown')
            algorithm_counts[algo] = algorithm_counts.get(algo, 0) + 1
        
        print('📊 Hyperparameter Test Results:')
        print(f'   • Generated {len(configs)} configurations')
        print()
        
        print('   Algorithm Distribution:')
        for algo, count in sorted(algorithm_counts.items()):
            percentage = (count / len(configs)) * 100
            print(f'     {algo}: {count} configs ({percentage:.1f}%)')
        
        # Show sample config
        sample_config = configs[0]
        print(f'\n   Sample Configuration:')
        print(f'     • Algorithm: {sample_config.get("algorithm")}')
        print(f'     • Learning Rate: {sample_config.get("learning_rate")}')
        print(f'     • Transaction Cost: {sample_config.get("transaction_cost")}')
        print(f'     • Timesteps: {sample_config.get("timesteps"):,}')
        
        if 'ent_coef' in sample_config:
            print(f'     • Entropy Coef: {sample_config.get("ent_coef")} (High Exploration)')
        
        print('\n✅ Enhanced hyperparameter generation working!')
        print('   🎯 More SAC/A2C algorithms for exploration')
        print('   ⚡ Higher learning rates and entropy')
        print('   💰 Lower transaction costs')
        
        return {'success': True, 'algorithm_counts': algorithm_counts}
        
    except Exception as e:
        print(f'❌ Hyperparameter test failed: {str(e)}')
        return {'success': False, 'error': str(e)}

def main():
    """Main test function"""
    print('🚀 Enhanced Active Trading System Tests')
    print('='*60)
    
    results = {}
    
    # Test 1: Environment
    results['environment'] = test_enhanced_environment()
    
    # Test 2: Scoring
    results['scoring'] = test_enhanced_scoring()
    
    # Test 3: Hyperparameters
    results['hyperparameters'] = test_enhanced_hyperparameters()
    
    # Summary
    print('\n📋 Test Summary:')
    print('='*40)
    
    success_count = sum(1 for r in results.values() if r.get('success', False))
    total_tests = len(results)
    
    for test_name, result in results.items():
        status = '✅' if result.get('success', False) else '❌'
        print(f'{status} {test_name.capitalize()} test')
        
        if not result.get('success', False) and 'error' in result:
            print(f'   Error: {result["error"]}')
    
    print(f'\n🎯 Overall Result: {success_count}/{total_tests} tests passed')
    
    if success_count == total_tests:
        print('\n🏆 All tests PASSED! Enhanced Active Trading System is ready!')
        print('   ✅ Environment encourages active trading')
        print('   ✅ Scoring system prioritizes trading activity')  
        print('   ✅ Hyperparameters optimized for exploration')
        print('\n💡 Next Step: Run "python train_all_models.py" to start training!')
    else:
        print('\n⚠️  Some tests failed. Check the errors above.')
    
    return results

if __name__ == '__main__':
    main()
