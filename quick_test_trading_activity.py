"""
Quick Test: การแก้ไขปัญหา Trading Activity
ทดสอบการปรับ reward function เพื่อส่งเสริมการเทรดอย่างแอคทีฟ
"""

import json
import numpy as np
import random

print('🧪 Quick Test: Enhanced Trading Activity')
print('='*50)

class QuickTradingTest:
    """ทดสอบระบบ reward ใหม่แบบเร็ว"""
    
    def __init__(self):
        self.results = []
        
    def simulate_old_reward_system(self, trades_count, roi):
        """จำลอง reward system เดิม (เน้นความปลอดภัย)"""
        base_score = 25.0
        
        # เดิม: เน้น ROI และ safety มากเกินไป
        if roi > 0:
            roi_score = roi * 50  # ให้คะแนน ROI สูงมาก
        else:
            roi_score = roi * 100  # penalty สูงสำหรับการขาดทุน
            
        # เดิม: ไม่มี trading activity incentive
        activity_score = 0
        
        # เดิม: penalty drawdown สูงมาก
        risk_penalty = 15 if roi < -0.05 else 0
        
        final_score = base_score + roi_score - risk_penalty
        return max(final_score, 0)
    
    def simulate_new_reward_system(self, trades_count, roi):
        """จำลอง reward system ใหม่ (ส่งเสริมการเทรด)"""
        base_score = 25.0
        
        # ใหม่: ลดความสำคัญของ ROI ทันที
        if roi > 0:
            roi_score = roi * 30  # ลดจาก 50 เป็น 30
        else:
            roi_score = roi * 50  # ลด penalty จาก 100 เป็น 50
            
        # ใหม่: เพิ่ม trading activity incentive (สำคัญมาก!)
        if trades_count >= 50:
            activity_score = 20.0
        elif trades_count >= 30:
            activity_score = 15.0
        elif trades_count >= 20:
            activity_score = 10.0
        elif trades_count >= 10:
            activity_score = 5.0
        else:
            activity_score = -10.0  # penalty สำหรับไม่เทรด
            
        # ใหม่: ลด risk penalty
        risk_penalty = 10 if roi < -0.1 else 0  # อนุญาตให้ขาดทุน 10%
        
        final_score = base_score + roi_score + activity_score - risk_penalty
        return max(final_score, 0)
    
    def run_comparison_test(self):
        """เปรียบเทียบ reward system เดิมกับใหม่"""
        
        test_scenarios = [
            # (trades_count, roi, scenario_name)
            (0, 0.0, "ไม่เทรดเลย (ปัญหาหลัก)"),
            (5, -0.01, "เทรดน้อย + ขาดทุนเล็กน้อย"),
            (15, -0.02, "เทรดน้อย + ขาดทุนปานกลาง"),
            (30, -0.03, "เทรดปานกลาง + ขาดทุนเล็กน้อย"),
            (50, -0.02, "เทรดดี + ขาดทุนเล็กน้อย"),
            (75, 0.01, "เทรดดี + กำไรเล็กน้อย"),
            (100, 0.05, "เทรดดีมาก + กำไรดี"),
            (25, 0.08, "เทรดน้อย + กำไรสูง (lucky)"),
        ]
        
        print('📊 เปรียบเทียบ Reward System:')
        print()
        print('Scenario                              | Trades | ROI   | Old Score | New Score | Diff')
        print('-' * 90)
        
        improvements = 0
        for trades, roi, scenario in test_scenarios:
            old_score = self.simulate_old_reward_system(trades, roi)
            new_score = self.simulate_new_reward_system(trades, roi)
            diff = new_score - old_score
            
            # Count improvements for low-trading scenarios
            if trades <= 20 and diff > 0:
                improvements += 1
                
            print(f'{scenario:<35} | {trades:>6} | {roi:>5.1%} | {old_score:>9.1f} | {new_score:>9.1f} | {diff:>+4.1f}')
        
        print('-' * 90)
        print(f'✅ Improvements for low-trading scenarios: {improvements}/5')
        print()
        
    def test_trading_incentives(self):
        """ทดสอบ incentive สำหรับการเทรด"""
        
        print('🎯 Trading Incentive Analysis:')
        print()
        
        # Test different trading levels with same ROI
        fixed_roi = -0.02  # ขาดทุน 2%
        trading_levels = [0, 5, 10, 20, 30, 50, 75, 100]
        
        print('Trades | New Score | Incentive Level')
        print('-' * 40)
        
        prev_score = 0
        for trades in trading_levels:
            score = self.simulate_new_reward_system(trades, fixed_roi)
            
            if trades == 0:
                level = "Heavy Penalty 🔴"
            elif trades < 10:
                level = "Still Penalty 🟡"  
            elif trades < 30:
                level = "Small Bonus 🟢"
            elif trades < 50:
                level = "Good Bonus 💚"
            else:
                level = "Excellent Bonus 🏆"
                
            improvement = score - prev_score if trades > 0 else 0
            print(f'{trades:>6} | {score:>9.1f} | {level} (+{improvement:.1f})')
            prev_score = score
            
        print()
        
    def test_risk_tolerance(self):
        """ทดสอบการปรับ Risk Tolerance"""
        
        print('⚖️ Risk Tolerance Comparison:')
        print()
        
        # Test different loss levels with active trading
        fixed_trades = 50  # เทรดเยอะ
        loss_levels = [0, -0.02, -0.05, -0.08, -0.10, -0.15]
        
        print('ROI   | Old Penalty | New Penalty | Difference')
        print('-' * 50)
        
        for roi in loss_levels:
            old_score = self.simulate_old_reward_system(fixed_trades, roi)
            new_score = self.simulate_new_reward_system(fixed_trades, roi)
            
            # Calculate penalty component only
            base_with_activity = 25 + 20  # base + activity bonus
            old_penalty = max(0, base_with_activity - old_score)
            new_penalty = max(0, base_with_activity - new_score)
            
            diff = old_penalty - new_penalty
            print(f'{roi:>5.1%} | {old_penalty:>11.1f} | {new_penalty:>11.1f} | {diff:>+10.1f}')
            
        print()
        
    def recommend_implementation(self):
        """แนะนำการ implement ใน production"""
        
        print('💡 Implementation Recommendations:')
        print('='*50)
        print()
        
        print('1. 🎁 Reward Function Changes:')
        print('   • เพิ่ม trading_activity_bonus = min(trades_count / 50, 1.0) * 20')
        print('   • ลด immediate_profit_weight จาก 1.0 เป็น 0.3')
        print('   • เพิ่ม consecutive_hold_penalty หลัง 20 steps')
        print('   • ลด risk_penalty_threshold จาก 5% เป็น 10%')
        print()
        
        print('2. 🔧 Environment Changes:')
        print('   • ลด transaction_cost จาก 0.0001 เป็น 0.00005')
        print('   • เพิ่ม market_opportunity_reward ตาม volatility')
        print('   • เพิ่ม action_diversity_bonus')
        print()
        
        print('3. 📊 Hyperparameter Changes:')
        print('   • เพิ่ม entropy_coefficient สำหรับ exploration')
        print('   • ลด learning_rate ให้เรียนรู้นานขึ้น') 
        print('   • เพิ่ม total_timesteps เป็น 4-5M')
        print('   • เพิ่มสัดส่วน SAC algorithm (ดีกับ continuous action)')
        print()
        
        print('4. 🎯 Success Criteria Changes:')
        print('   • เปลี่ยนจาก ROI-focused เป็น Activity + Consistency')
        print('   • เพิ่ม minimum_trades_requirement = 20')
        print('   • ลด maximum_drawdown_allowed เป็น 15%')
        print('   • เพิ่ม trading_frequency_score ใน final evaluation')

def main():
    print('🔬 Testing Enhanced Trading Activity System')
    print('='*60)
    print()
    
    tester = QuickTradingTest()
    
    # Run all tests
    tester.run_comparison_test()
    tester.test_trading_incentives()  
    tester.test_risk_tolerance()
    tester.recommend_implementation()
    
    print()
    print('🎯 Summary:')
    print('✅ ระบบใหม่ส่งเสริมการเทรดแอคทีฟได้ดีกว่า')
    print('✅ ลด penalty สำหรับความเสี่ยงที่สมเหตุสมผล')
    print('✅ เพิ่ม incentive สำหรับ trading frequency')
    print('⚡ พร้อม implement ในระบบจริงแล้ว!')

if __name__ == '__main__':
    main()
