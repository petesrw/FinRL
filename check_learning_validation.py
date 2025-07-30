#!/usr/bin/env python3
"""
🔍 Learning Validation Tool
ตรวจสอบว่า AI model ได้เรียนรู้จริงๆ หรือแค่สุ่ม actions
"""

import numpy as np
import torch
import json
import os
import matplotlib.pyplot as plt
from datetime import datetime
from stable_baselines3 import PPO, SAC, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from data_manager import load_data
from train_all_models import AdvancedForexEnv
import random
import time

def analyze_model_weights(model_path):
    """วิเคราะห์ weights ของ model"""
    print(f"🔍 Analyzing model weights: {model_path}")
    
    try:
        # โหลด model
        if 'ppo' in model_path.lower() or 'bronze' in model_path.lower():
            model = PPO.load(model_path)
        elif 'sac' in model_path.lower():
            model = SAC.load(model_path)
        else:
            model = PPO.load(model_path)  # Default to PPO
        
        # ดึง weights จาก policy network
        policy_weights = []
        
        if hasattr(model.policy, 'mlp_extractor'):
            # PPO/A2C structure
            for name, param in model.policy.mlp_extractor.named_parameters():
                if 'weight' in name:
                    weights = param.detach().cpu().numpy().flatten()
                    policy_weights.extend(weights)
        elif hasattr(model.policy, 'actor'):
            # SAC structure
            for name, param in model.policy.actor.named_parameters():
                if 'weight' in name:
                    weights = param.detach().cpu().numpy().flatten()
                    policy_weights.extend(weights)
        
        if policy_weights:
            weights_array = np.array(policy_weights)
            
            analysis = {
                'total_weights': len(weights_array),
                'mean_weight': float(np.mean(weights_array)),
                'std_weight': float(np.std(weights_array)),
                'min_weight': float(np.min(weights_array)),
                'max_weight': float(np.max(weights_array)),
                'weight_range': float(np.max(weights_array) - np.min(weights_array)),
                'zero_weights_pct': float(np.sum(np.abs(weights_array) < 1e-6) / len(weights_array) * 100),
                'large_weights_pct': float(np.sum(np.abs(weights_array) > 1.0) / len(weights_array) * 100)
            }
            
            print(f"   📊 Weight Statistics:")
            print(f"      Total weights: {analysis['total_weights']:,}")
            print(f"      Mean: {analysis['mean_weight']:.6f}")
            print(f"      Std: {analysis['std_weight']:.6f}")
            print(f"      Range: [{analysis['min_weight']:.6f}, {analysis['max_weight']:.6f}]")
            print(f"      Zero weights: {analysis['zero_weights_pct']:.2f}%")
            print(f"      Large weights (>1): {analysis['large_weights_pct']:.2f}%")
            
            # การประเมิน learning
            if analysis['std_weight'] < 0.01:
                print(f"   ⚠️ WARNING: Very low weight variance - possible poor learning")
            elif analysis['std_weight'] > 2.0:
                print(f"   ⚠️ WARNING: Very high weight variance - possible instability")
            else:
                print(f"   ✅ Weight variance looks healthy")
            
            if analysis['zero_weights_pct'] > 50:
                print(f"   ⚠️ WARNING: Too many zero weights - possible dead neurons")
            
            return analysis
        else:
            print(f"   ❌ Could not extract weights from model")
            return None
            
    except Exception as e:
        print(f"   ❌ Error analyzing weights: {e}")
        return None

def test_random_vs_learned_behavior(model_path, data, test_episodes=5):
    """เปรียบเทียบ behavior ของ trained model กับ random actions"""
    print(f"\n🎲 Testing Random vs Learned Behavior...")
    
    try:
        # โหลด model
        if 'ppo' in model_path.lower() or 'bronze' in model_path.lower():
            model = PPO.load(model_path)
        elif 'sac' in model_path.lower():
            model = SAC.load(model_path)
        else:
            model = PPO.load(model_path)
        
        # เตรียม environment
        env = AdvancedForexEnv(
            data.tail(2000), 
            symbol='EURUSD',
            lookback_window=50,
            transaction_cost=0
        )
        
        # Test learned model
        learned_results = []
        for episode in range(test_episodes):
            obs, _ = env.reset()
            done = False
            episode_actions = []
            episode_rewards = []
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _, info = env.step(action)
                episode_actions.append(action[0] if isinstance(action, np.ndarray) else action)
                episode_rewards.append(reward)
            
            learned_results.append({
                'actions': episode_actions,
                'rewards': episode_rewards,
                'total_reward': sum(episode_rewards),
                'final_balance': env.balance,
                'total_trades': env.total_trades
            })
        
        # Test random actions
        random_results = []
        for episode in range(test_episodes):
            obs, _ = env.reset()
            done = False
            episode_actions = []
            episode_rewards = []
            
            while not done:
                # Random action in range [-1, 1] (same as trained model)
                action = np.random.uniform(-1, 1)
                obs, reward, done, _, info = env.step(action)
                episode_actions.append(action)
                episode_rewards.append(reward)
            
            random_results.append({
                'actions': episode_actions,
                'rewards': episode_rewards,
                'total_reward': sum(episode_rewards),
                'final_balance': env.balance,
                'total_trades': env.total_trades
            })
        
        # วิเคราะห์ผลลัพธ์
        learned_stats = {
            'avg_reward': np.mean([r['total_reward'] for r in learned_results]),
            'avg_balance': np.mean([r['final_balance'] for r in learned_results]),
            'avg_trades': np.mean([r['total_trades'] for r in learned_results]),
            'action_std': np.std([action for r in learned_results for action in r['actions']])
        }
        
        random_stats = {
            'avg_reward': np.mean([r['total_reward'] for r in random_results]),
            'avg_balance': np.mean([r['final_balance'] for r in random_results]),
            'avg_trades': np.mean([r['total_trades'] for r in random_results]),
            'action_std': np.std([action for r in random_results for action in r['actions']])
        }
        
        print(f"\n📊 Performance Comparison ({test_episodes} episodes):")
        print(f"   🤖 Learned Model:")
        print(f"      Avg Reward: {learned_stats['avg_reward']:.2f}")
        print(f"      Avg Balance: {learned_stats['avg_balance']:.2f}")
        print(f"      Avg Trades: {learned_stats['avg_trades']:.1f}")
        print(f"      Action Std: {learned_stats['action_std']:.4f}")
        
        print(f"   🎲 Random Actions:")
        print(f"      Avg Reward: {random_stats['avg_reward']:.2f}")
        print(f"      Avg Balance: {random_stats['avg_balance']:.2f}")
        print(f"      Avg Trades: {random_stats['avg_trades']:.1f}")
        print(f"      Action Std: {random_stats['action_std']:.4f}")
        
        # การประเมิน
        reward_improvement = learned_stats['avg_reward'] - random_stats['avg_reward']
        balance_improvement = learned_stats['avg_balance'] - random_stats['avg_balance']
        
        print(f"\n🎯 Learning Assessment:")
        print(f"   Reward Improvement: {reward_improvement:+.2f}")
        print(f"   Balance Improvement: {balance_improvement:+.2f}")
        
        if reward_improvement > 10:
            print(f"   ✅ STRONG LEARNING: Model significantly outperforms random")
        elif reward_improvement > 0:
            print(f"   ✅ MODERATE LEARNING: Model shows some improvement")
        elif reward_improvement > -10:
            print(f"   ⚠️ WEAK LEARNING: Model similar to random")
        else:
            print(f"   ❌ NO LEARNING: Model worse than random")
        
        # Action distribution analysis
        learned_actions = [action for r in learned_results for action in r['actions']]
        random_actions = [action for r in random_results for action in r['actions']]
        
        print(f"\n📈 Action Distribution Analysis:")
        print(f"   🤖 Learned Actions:")
        print(f"      Mean: {np.mean(learned_actions):.4f}")
        print(f"      Std: {np.std(learned_actions):.4f}")
        print(f"      Range: [{np.min(learned_actions):.4f}, {np.max(learned_actions):.4f}]")
        
        print(f"   🎲 Random Actions:")
        print(f"      Mean: {np.mean(random_actions):.4f}")
        print(f"      Std: {np.std(random_actions):.4f}")
        print(f"      Range: [{np.min(random_actions):.4f}, {np.max(random_actions):.4f}]")
        
        # ตรวจสอบ bias
        if abs(np.mean(learned_actions)) > 0.3:
            print(f"   ⚠️ ACTION BIAS detected: Model favors {'BUY' if np.mean(learned_actions) > 0 else 'SELL'}")
        
        if np.std(learned_actions) < 0.1:
            print(f"   ⚠️ LOW ACTION VARIANCE: Model may be stuck in limited action range")
        
        return {
            'learned_stats': learned_stats,
            'random_stats': random_stats,
            'reward_improvement': reward_improvement,
            'balance_improvement': balance_improvement
        }
        
    except Exception as e:
        print(f"   ❌ Error testing behavior: {e}")
        return None

def check_training_progression():
    """ตรวจสอบ progression ของการ training จาก logs"""
    print(f"\n📈 Checking Training Progression...")
    
    log_file = "training_logs/async_logs/eurusd_async_training.json"
    
    if not os.path.exists(log_file):
        print(f"   ❌ No training log found: {log_file}")
        return
    
    try:
        with open(log_file, 'r') as f:
            data = json.load(f)
        
        all_models = []
        for batch in data:
            if 'results' in batch:
                for result in batch['results']:
                    if result.get('success', False):
                        all_models.append(result)
        
        if len(all_models) < 2:
            print(f"   ⚠️ Not enough training data to analyze progression")
            return
        
        # วิเคราะห์ progression
        scores = [m['score'] for m in all_models]
        returns = [m['metrics']['total_return'] for m in all_models]
        win_rates = [m['metrics']['win_rate'] for m in all_models]
        
        print(f"   📊 Training Progression Analysis:")
        print(f"      Total Models: {len(all_models)}")
        print(f"      Score Range: {min(scores):.1f} - {max(scores):.1f}")
        print(f"      Return Range: {min(returns):.4f} - {max(returns):.4f}")
        print(f"      Win Rate Range: {min(win_rates):.3f} - {max(win_rates):.3f}")
        
        # ตรวจสอบการปรับปรุง
        if len(scores) >= 3:
            recent_scores = scores[-3:]
            early_scores = scores[:3]
            
            recent_avg = np.mean(recent_scores)
            early_avg = np.mean(early_scores)
            
            improvement = recent_avg - early_avg
            print(f"      Score Improvement: {improvement:+.1f}")
            
            if improvement > 5:
                print(f"   ✅ LEARNING DETECTED: Scores improving over time")
            elif improvement > 0:
                print(f"   ✅ SLOW LEARNING: Slight improvement detected")
            else:
                print(f"   ⚠️ NO IMPROVEMENT: Scores not improving")
        
        # ค้นหา best model
        best_model = max(all_models, key=lambda x: x['score'])
        print(f"\n🏆 Best Model Found:")
        print(f"   Score: {best_model['score']:.1f}")
        print(f"   Tier: {best_model['tier'].upper()}")
        print(f"   Return: {best_model['metrics']['total_return']:.4f}")
        print(f"   Win Rate: {best_model['metrics']['win_rate']:.3f}")
        print(f"   Trades: {best_model['metrics']['total_trades']}")
        
        return best_model
        
    except Exception as e:
        print(f"   ❌ Error analyzing progression: {e}")
        return None

def test_model_consistency(model_path, data, num_runs=3):
    """ทดสอบความสม่ำเสมอของ model"""
    print(f"\n🔄 Testing Model Consistency ({num_runs} runs)...")
    
    try:
        if 'ppo' in model_path.lower() or 'bronze' in model_path.lower():
            model = PPO.load(model_path)
        elif 'sac' in model_path.lower():
            model = SAC.load(model_path)
        else:
            model = PPO.load(model_path)
        
        env = AdvancedForexEnv(
            data.tail(1000), 
            symbol='EURUSD',
            lookback_window=50,
            transaction_cost=0
        )
        
        run_results = []
        
        for run in range(num_runs):
            obs, _ = env.reset()
            done = False
            actions_in_run = []
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _, info = env.step(action)
                actions_in_run.append(action[0] if isinstance(action, np.ndarray) else action)
            
            run_results.append({
                'actions': actions_in_run,
                'final_balance': env.balance,
                'total_trades': env.total_trades
            })
        
        # วิเคราะห์ความสม่ำเสมอ
        balances = [r['final_balance'] for r in run_results]
        trades = [r['total_trades'] for r in run_results]
        
        balance_std = np.std(balances)
        trades_std = np.std(trades)
        
        print(f"   📊 Consistency Analysis:")
        print(f"      Balance Std: {balance_std:.2f}")
        print(f"      Trades Std: {trades_std:.2f}")
        
        if balance_std < 50 and trades_std < 5:
            print(f"   ✅ HIGH CONSISTENCY: Model behaves predictably")
        elif balance_std < 200:
            print(f"   ✅ MODERATE CONSISTENCY: Some variation in results")
        else:
            print(f"   ⚠️ LOW CONSISTENCY: High variation - possible randomness")
        
        return {
            'balance_std': balance_std,
            'trades_std': trades_std,
            'results': run_results
        }
        
    except Exception as e:
        print(f"   ❌ Error testing consistency: {e}")
        return None

def main():
    """Main validation function"""
    print("🔍 AI MODEL LEARNING VALIDATION")
    print("="*60)
    print("ตรวจสอบว่า AI model ได้เรียนรู้จริงๆ หรือแค่สุ่ม actions")
    print()
    
    # หา model ที่ดีที่สุด
    model_dirs = [
        'models/bronze',
        'models/silver', 
        'models/gold',
        'models/diamond'
    ]
    
    best_model_path = None
    for model_dir in model_dirs:
        if os.path.exists(model_dir):
            models = [f for f in os.listdir(model_dir) if f.endswith('.zip')]
            if models:
                # เลือก model ล่าสุด
                latest_model = max(models, key=lambda x: os.path.getctime(os.path.join(model_dir, x)))
                best_model_path = os.path.join(model_dir, latest_model)
                break
    
    if not best_model_path:
        print("❌ No trained models found!")
        print("   กรุณา train model ก่อนใช้เครื่องมือนี้")
        return
    
    print(f"🤖 Found model: {best_model_path}")
    
    # โหลดข้อมูล
    print(f"\n📊 Loading data...")
    data = load_data('EURUSD')
    if data is None:
        print("❌ Failed to load data")
        return
    
    print(f"✅ Data loaded: {len(data)} records")
    
    # รันการทดสอบทั้งหมด
    print("\n" + "="*60)
    
    # 1. วิเคราะห์ weights
    weight_analysis = analyze_model_weights(best_model_path)
    
    # 2. เปรียบเทียบกับ random
    behavior_analysis = test_random_vs_learned_behavior(best_model_path, data)
    
    # 3. ตรวจสอบ progression
    progression_analysis = check_training_progression()
    
    # 4. ทดสอบความสม่ำเสมอ
    consistency_analysis = test_model_consistency(best_model_path, data)
    
    # สรุปผลลัพธ์
    print("\n" + "="*60)
    print("📋 FINAL LEARNING VALIDATION SUMMARY")
    print("="*60)
    
    learning_indicators = []
    
    if weight_analysis:
        if 0.01 <= weight_analysis['std_weight'] <= 2.0:
            learning_indicators.append("✅ Healthy weight distribution")
        else:
            learning_indicators.append("⚠️ Problematic weight distribution")
    
    if behavior_analysis:
        if behavior_analysis['reward_improvement'] > 5:
            learning_indicators.append("✅ Strong performance vs random")
        elif behavior_analysis['reward_improvement'] > 0:
            learning_indicators.append("✅ Some improvement vs random")
        else:
            learning_indicators.append("❌ No improvement vs random")
    
    if consistency_analysis:
        if consistency_analysis['balance_std'] < 200:
            learning_indicators.append("✅ Consistent behavior")
        else:
            learning_indicators.append("⚠️ Inconsistent behavior")
    
    print("\n🎯 Learning Indicators:")
    for indicator in learning_indicators:
        print(f"   {indicator}")
    
    # คำแนะนำ
    if len([i for i in learning_indicators if i.startswith("✅")]) >= 2:
        print(f"\n🎉 CONCLUSION: Model shows GOOD LEARNING")
        print(f"   🎯 Model ได้เรียนรู้จริงๆ ไม่ใช่แค่สุ่ม")
        print(f"   🚀 สามารถใช้ใน live trading ได้")
    else:
        print(f"\n⚠️ CONCLUSION: Learning is QUESTIONABLE")
        print(f"   🔄 ควร retrain model ด้วย hyperparameters ใหม่")
        print(f"   📈 เพิ่ม exploration หรือ training time")
    
    print(f"\n💡 Recommendations:")
    print(f"   1. หาก learning ดี: ทดสอบกับ demo account")
    print(f"   2. หาก learning แย่: ปรับ hyperparameters")
    print(f"   3. ตรวจสอบ bias detection ใน live trading")
    print(f"   4. Monitor ผลลัพธ์ระยะยาว")

if __name__ == "__main__":
    main()
