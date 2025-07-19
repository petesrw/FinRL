#!/usr/bin/env python3
"""
Test the new scoring criteria
"""

import sys
sys.path.append('.')

# Mock the dependencies
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
from datetime import datetime
import json

class MockAdaptiveTrainer:
    """Mock trainer for testing scoring"""
    
    def __init__(self):
        # Realistic Excellence targets for Forex Trading (adjusted score thresholds)
        self.targets = {
            'bronze': {'win_rate': 0.55, 'profit_factor': 1.5, 'max_drawdown': 0.20, 'score': 45},
            'silver': {'win_rate': 0.60, 'profit_factor': 1.8, 'max_drawdown': 0.18, 'score': 55},
            'gold': {'win_rate': 0.65, 'profit_factor': 2.2, 'max_drawdown': 0.15, 'score': 70},
            'diamond': {'win_rate': 0.70, 'profit_factor': 2.5, 'max_drawdown': 0.12, 'score': 85}
        }
    
    def calculate_score(self, metrics):
        """Calculate overall performance score with realistic weighting for Forex"""
        win_rate = metrics.get('win_rate', 0)
        profit_factor = metrics.get('profit_factor', 0)
        max_drawdown = metrics.get('max_drawdown', 1)
        sharpe_ratio = metrics.get('sharpe_ratio', 0)
        total_return = metrics.get('total_return', 0)
        
        # Base scores (0-100 scale)
        win_rate_score = win_rate * 100  # Direct conversion to percentage
        
        # Profit factor: 1.0=0, 1.5=25, 2.0=50, 2.5=75, 3.0+=100
        pf_score = min((profit_factor - 1.0) * 50, 100) if profit_factor >= 1.0 else 0
        
        # Drawdown penalty: 0%=100, 5%=90, 10%=80, 15%=70, 20%=60, 25%=50, 30%+=0
        dd_score = max(0, 100 - (max_drawdown * 100 * 3.33))
        
        # Sharpe ratio: 0=0, 0.5=25, 1.0=50, 1.5=75, 2.0+=100
        sharpe_score = min(sharpe_ratio * 50, 100)
        
        # Total return component: negative return penalty, positive return bonus
        if total_return < 0:
            return_component = total_return * 100  # Penalty for negative returns
        else:
            return_component = min(total_return * 50, 25)  # Bonus up to 25 points
        
        # Main score calculation (balanced weights)
        main_score = (
            win_rate_score * 0.35 +      # 35% weight on win rate
            pf_score * 0.30 +            # 30% weight on profit factor  
            dd_score * 0.25 +            # 25% weight on drawdown control
            sharpe_score * 0.10          # 10% weight on Sharpe ratio
        )
        
        # Add return component (can be negative)
        final_score = main_score + return_component
        
        # Ensure realistic minimum for poor performance
        if win_rate < 0.30 or profit_factor < 1.0 or max_drawdown > 0.50:
            final_score = min(final_score, 30)  # Cap very poor performance
        
        return max(0, min(final_score, 100))  # Ensure score is between 0-100
    
    def get_tier(self, metrics):
        """Determine performance tier with stricter win rate requirements"""
        win_rate = metrics.get('win_rate', 0)
        profit_factor = metrics.get('profit_factor', 0)
        max_drawdown = metrics.get('max_drawdown', 1)
        score = self.calculate_score(metrics)
        
        # All tiers require minimum win rate AND must pass ALL criteria
        if (win_rate >= self.targets['diamond']['win_rate'] and 
            profit_factor >= self.targets['diamond']['profit_factor'] and 
            max_drawdown <= self.targets['diamond']['max_drawdown'] and
            score >= self.targets['diamond']['score']):
            return 'diamond', '💎'
        elif (win_rate >= self.targets['gold']['win_rate'] and 
              profit_factor >= self.targets['gold']['profit_factor'] and 
              max_drawdown <= self.targets['gold']['max_drawdown'] and
              score >= self.targets['gold']['score']):
            return 'gold', '🥇'
        elif (win_rate >= self.targets['silver']['win_rate'] and 
              profit_factor >= self.targets['silver']['profit_factor'] and 
              max_drawdown <= self.targets['silver']['max_drawdown'] and
              score >= self.targets['silver']['score']):
            return 'silver', '🥈'
        elif (win_rate >= self.targets['bronze']['win_rate'] and 
              profit_factor >= self.targets['bronze']['profit_factor'] and 
              max_drawdown <= self.targets['bronze']['max_drawdown'] and
              score >= self.targets['bronze']['score']):
            return 'bronze', '🥉'
        else:
            return 'none', '❌'

def main():
    """Test scoring system"""
    print('🧪 ทดสอบเกณฑ์การให้คะแนนใหม่ (ปรับปรุงแล้ว)')
    print('='*60)
    
    trainer = MockAdaptiveTrainer()
    
    # Test cases with various performance levels
    test_cases = [
        {
            'name': 'Bronze Target',
            'metrics': {'win_rate': 0.55, 'profit_factor': 1.5, 'max_drawdown': 0.20, 'sharpe_ratio': 1.0, 'total_return': 0.10}
        },
        {
            'name': 'Silver Target', 
            'metrics': {'win_rate': 0.60, 'profit_factor': 1.8, 'max_drawdown': 0.18, 'sharpe_ratio': 1.2, 'total_return': 0.15}
        },
        {
            'name': 'Gold Target',
            'metrics': {'win_rate': 0.65, 'profit_factor': 2.2, 'max_drawdown': 0.15, 'sharpe_ratio': 1.5, 'total_return': 0.25}
        },
        {
            'name': 'Diamond Target',
            'metrics': {'win_rate': 0.70, 'profit_factor': 2.5, 'max_drawdown': 0.12, 'sharpe_ratio': 2.0, 'total_return': 0.30}
        },
        {
            'name': 'Poor Performance',
            'metrics': {'win_rate': 0.45, 'profit_factor': 1.2, 'max_drawdown': 0.30, 'sharpe_ratio': 0.5, 'total_return': -0.05}
        },
        {
            'name': 'Excellent Performance', 
            'metrics': {'win_rate': 0.75, 'profit_factor': 3.0, 'max_drawdown': 0.08, 'sharpe_ratio': 2.5, 'total_return': 0.45}
        },
        {
            'name': 'High Win Rate, Poor Risk',
            'metrics': {'win_rate': 0.80, 'profit_factor': 1.1, 'max_drawdown': 0.40, 'sharpe_ratio': 0.2, 'total_return': 0.02}
        },
        {
            'name': 'Low Win Rate, Good Risk',
            'metrics': {'win_rate': 0.40, 'profit_factor': 2.8, 'max_drawdown': 0.05, 'sharpe_ratio': 1.8, 'total_return': 0.35}
        }
    ]
    
    print(f"📋 เกณฑ์ Tier (ปรับปรุงแล้ว):")
    for tier, criteria in trainer.targets.items():
        print(f"  {tier.upper()}: Win Rate {criteria['win_rate']:.0%}, PF {criteria['profit_factor']:.1f}, DD {criteria['max_drawdown']:.0%}, Score {criteria['score']}")
    print()
    
    for i, test_case in enumerate(test_cases):
        metrics = test_case['metrics']
        score = trainer.calculate_score(metrics)
        tier, emoji = trainer.get_tier(metrics)
        
        print(f"Test {i+1}: {test_case['name']}")
        print(f"  {emoji} Tier: {tier.upper()} | Score: {score:.1f}")
        print(f"  📈 Win Rate: {metrics['win_rate']:.1%}")
        print(f"  💰 Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"  📉 Max Drawdown: {metrics['max_drawdown']:.1%}")
        print(f"  🎯 Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"  💵 Total Return: {metrics['total_return']:.1%}")
        print()
    
    print("✅ เกณฑ์การให้คะแนนปรับปรุงเรียบร้อย!")

if __name__ == "__main__":
    main()
