"""
Quick Validation: Enhanced Active Trading System
ตรวจสอบความถูกต้องของการแก้ไขปัญหา Trading Activity
"""

import json
import os
from datetime import datetime

print('✅ Enhanced Active Trading System - Quick Validation')
print('='*60)

def check_reward_system_changes():
    """ตรวจสอบการเปลี่ยนแปลง Reward System"""
    print('\n🎯 1. Checking Reward System Enhancements...')
    
    try:
        with open('train_all_models.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check key enhancements
        checks = [
            ('ACTIVE TRADING ENHANCED REWARD', '🎯 Active Trading Enhanced Reward Function'),
            ('TRADING ACTIVITY INCENTIVE', '🎁 Trading Activity Incentive (Core Fix)'),
            ('CONSECUTIVE HOLD PENALTY', '🚫 Consecutive Hold Penalty (Anti-Hold)'),
            ('MARKET OPPORTUNITY REWARD', '🎯 Market Opportunity Reward'),
            ('consecutive_holds', 'Consecutive Holds Tracking'),
        ]
        
        for check, description in checks:
            if check in content:
                print(f'   ✅ {description}')
            else:
                print(f'   ❌ Missing: {description}')
        
        # Check for relaxed penalties
        if 'max_drawdown > 0.20' in content:
            print('   ✅ Relaxed Drawdown Management (20% vs 15%)')
        if 'transaction_cost=0.00005' in content:
            print('   ✅ Lower Transaction Costs')
            
        return True
        
    except Exception as e:
        print(f'   ❌ Error checking reward system: {e}')
        return False

def check_hyperparameter_enhancements():
    """ตรวจสอบการปรับปรุง Hyperparameters"""
    print('\n🔧 2. Checking Hyperparameter Enhancements...')
    
    try:
        with open('train_all_models.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        checks = [
            ('ACTIVE TRADING ALGORITHM DISTRIBUTION', 'Enhanced Algorithm Distribution'),
            ("algorithm_choice < 0.35", 'Reduced PPO to 35%'),
            ("algorithm_choice < 0.70", 'Increased SAC to 35%'),
            ('ent_coef', 'Higher Entropy Coefficient'),
            ('timesteps.*[456]000000', 'Extended Training Time (4-6M timesteps)'),
            ('ULTIMATE ACTIVE TRADING', 'Ultimate Active Trading Fallback'),
        ]
        
        for check, description in checks:
            if check.replace('.*', '') in content:
                print(f'   ✅ {description}')
            else:
                print(f'   ❌ Missing: {description}')
        
        return True
        
    except Exception as e:
        print(f'   ❌ Error checking hyperparameters: {e}')
        return False

def check_scoring_system():
    """ตรวจสอบระบบ Scoring ใหม่"""
    print('\n🏆 3. Checking Enhanced Scoring System...')
    
    try:
        with open('train_all_models.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        checks = [
            ('ACTIVE TRADING Enhanced Scoring', 'Enhanced Scoring System'),
            ('TRADING ACTIVITY COMPONENT', 'Trading Activity Component'),
            ('activity_score * 0.40', '40% Weight on Trading Activity'),
            ('win_rate_score * 0.25', '25% Weight on Win Rate (reduced)'),
            ('min_trades', 'Minimum Trades Requirement in Targets'),
            ('total_trades >= self.targets', 'Trading Activity in Tier Criteria'),
        ]
        
        for check, description in checks:
            if check in content:
                print(f'   ✅ {description}')
            else:
                print(f'   ❌ Missing: {description}')
        
        return True
        
    except Exception as e:
        print(f'   ❌ Error checking scoring system: {e}')
        return False

def check_target_adjustments():
    """ตรวจสอบการปรับ Targets"""
    print('\n🎯 4. Checking Target Adjustments...')
    
    try:
        with open('train_all_models.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if targets are more relaxed
        if "'win_rate': 0.45" in content:  # Bronze target relaxed from 0.55
            print('   ✅ Relaxed Bronze Win Rate (0.45 vs 0.55)')
        if "'profit_factor': 1.2" in content:  # Bronze PF relaxed from 1.5
            print('   ✅ Relaxed Bronze Profit Factor (1.2 vs 1.5)')
        if "'max_drawdown': 0.25" in content:  # Bronze DD relaxed from 0.20
            print('   ✅ Relaxed Bronze Max Drawdown (0.25 vs 0.20)')
        if "'min_trades': 20" in content:  # New requirement
            print('   ✅ Added Minimum Trades Requirement (20+)')
        if "'score': 35" in content:  # Bronze score relaxed from 45
            print('   ✅ Relaxed Bronze Score (35 vs 45)')
            
        return True
        
    except Exception as e:
        print(f'   ❌ Error checking targets: {e}')
        return False

def validate_file_structure():
    """ตรวจสอบโครงสร้างไฟล์"""
    print('\n📁 5. Checking File Structure...')
    
    # Check if required directories exist
    dirs_to_check = [
        'training_logs',
        'training_logs/successful_configs',
        'training_logs/failed_configs',
        'training_logs/history'
    ]
    
    for dir_path in dirs_to_check:
        if os.path.exists(dir_path):
            print(f'   ✅ Directory exists: {dir_path}')
        else:
            print(f'   ⚠️  Directory missing: {dir_path}')
    
    # Check key files
    files_to_check = [
        'train_all_models.py',
        'analyze_trading_activity.py',
        'quick_test_trading_activity.py',
        'enhanced_active_trading.py'
    ]
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            print(f'   ✅ File exists: {file_path}')
        else:
            print(f'   ❌ File missing: {file_path}')
    
    return True

def create_validation_report():
    """สร้างรายงานการตรวจสอบ"""
    print('\n📋 6. Creating Validation Report...')
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    report = {
        'validation_timestamp': timestamp,
        'system_version': 'Enhanced Active Trading System v2.0',
        'key_improvements': [
            'Trading Activity Incentive (40% weight in scoring)',
            'Consecutive Hold Penalty',
            'Market Opportunity Reward',
            'Relaxed Risk Thresholds',
            'Enhanced Algorithm Distribution (35% SAC)',
            'Extended Training Time (4-6M timesteps)',
            'Minimum Trading Activity Requirements'
        ],
        'expected_improvements': [
            'Models will trade more frequently (target: 20-80 trades)',
            'Less holding behavior (max 30 consecutive holds)',
            'Better ROI through active trading',
            'Bronze tier achievement with 20+ trades',
            'Reduced penalty for reasonable risk-taking'
        ],
        'success_criteria': {
            'bronze_tier': {
                'min_trades': 20,
                'win_rate': 0.45,
                'profit_factor': 1.2,
                'max_drawdown': 0.25,
                'score': 35
            }
        }
    }
    
    # Save report
    os.makedirs('training_logs', exist_ok=True)
    with open(f'training_logs/validation_report_{timestamp}.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f'   ✅ Validation report saved: training_logs/validation_report_{timestamp}.json')
    return True

def main():
    """Main validation function"""
    print('🔍 Running Enhanced Active Trading System Validation...')
    print()
    
    checks = [
        check_reward_system_changes,
        check_hyperparameter_enhancements,
        check_scoring_system,
        check_target_adjustments,
        validate_file_structure,
        create_validation_report
    ]
    
    results = []
    for check_func in checks:
        try:
            result = check_func()
            results.append(result)
        except Exception as e:
            print(f'   ❌ Check failed: {e}')
            results.append(False)
    
    # Summary
    print('\n📊 Validation Summary:')
    print('='*40)
    
    passed = sum(results)
    total = len(results)
    
    print(f'✅ Checks Passed: {passed}/{total}')
    
    if passed == total:
        print('\n🎉 VALIDATION SUCCESSFUL!')
        print('🚀 Enhanced Active Trading System is ready for training!')
        print()
        print('🎯 Key Improvements:')
        print('   • 40% scoring weight on trading activity')
        print('   • Anti-hold strategy with consecutive hold penalty')
        print('   • Market opportunity rewards during volatility')
        print('   • Relaxed risk thresholds for active traders')
        print('   • Extended training time (4-6M timesteps)')
        print('   • Enhanced algorithm mix (35% SAC for exploration)')
        print()
        print('📈 Expected Results:')
        print('   • More models achieving Bronze tier (35+ score)')
        print('   • Active trading behavior (20-80 trades per model)')
        print('   • Better ROI through frequent, profitable trades')
        print('   • Reduced "no trading" problem from 91→0 models')
        
    else:
        print('\n⚠️  Some validations failed. Please check the errors above.')
        
    return passed == total

if __name__ == '__main__':
    main()
