#!/usr/bin/env python3
"""
🔍 Model Bias Validator
ตรวจสอบ bias ในโมเดลที่บันทึกใน models/ directory
"""

import os
import sys
import glob
import json
import numpy as np
import pandas as pd
import shutil
from datetime import datetime
from stable_baselines3 import PPO, SAC, A2C
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Import environment และ classes ที่จำเป็น
try:
    from forex_rl_simple import SimpleForexBot
    from mt5_trading_bot import ModelLoader, TradingBot
    HAS_FOREX_MODULES = True
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("💡 Please ensure all required modules are available")
    HAS_FOREX_MODULES = False

class ObservationShapeWrapper:
    """Wrapper เพื่อปรับ observation shape ให้ตรงกับที่โมเดลต้องการ"""
    
    def __init__(self, env, target_shape):
        self.env = env
        self.target_shape = target_shape
        
        # Copy original observation space attributes if available
        try:
            self.observation_space = type('ObservationSpace', (), {})()
            self.observation_space.shape = target_shape
            if hasattr(env.observation_space, 'low'):
                self.observation_space.low = np.full(target_shape, -np.inf, dtype=np.float32)
                self.observation_space.high = np.full(target_shape, np.inf, dtype=np.float32)
            print(f"   🔧 ObservationWrapper: {env.observation_space.shape} -> {target_shape}")
        except Exception as e:
            print(f"   ⚠️ Warning creating observation space: {e}")
            self.observation_space = env.observation_space
    
    def reset(self, **kwargs):
        """Reset environment and adjust observation"""
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
            return self.observation(obs), info
        else:
            # Fallback for older gym versions
            return self.observation(result), {}
    
    def step(self, action):
        """Step environment and adjust observation"""
        result = self.env.step(action)
        if len(result) == 5:  # New format: obs, reward, done, truncated, info
            obs, reward, done, truncated, info = result
            return self.observation(obs), reward, done, truncated, info
        elif len(result) == 4:  # Old format: obs, reward, done, info
            obs, reward, done, info = result
            return self.observation(obs), reward, done, False, info
    
    def observation(self, obs):
        """ปรับ observation ให้ตรงกับ target shape"""
        try:
            # แปลงเป็น numpy array ก่อน
            if not isinstance(obs, np.ndarray):
                obs = np.array(obs, dtype=np.float32)
            
            current_shape = obs.shape
            
            if current_shape == self.target_shape:
                return obs
            
            # ถ้าเป็น 1D target
            if len(self.target_shape) == 1:
                target_size = self.target_shape[0]
                
                if len(current_shape) == 1:
                    # 1D -> 1D: resize
                    if current_shape[0] > target_size:
                        # ตัดส่วนเกิน
                        return obs[:target_size]
                    elif current_shape[0] < target_size:
                        # เพิ่ม padding
                        padding = np.zeros(target_size - current_shape[0])
                        return np.concatenate([obs, padding])
                    else:
                        return obs
                        
                elif len(current_shape) == 2:
                    # 2D -> 1D: flatten แล้ว resize
                    flattened = obs.flatten()
                    if len(flattened) > target_size:
                        return flattened[:target_size]
                    elif len(flattened) < target_size:
                        padding = np.zeros(target_size - len(flattened))
                        return np.concatenate([flattened, padding])
                    else:
                        return flattened
            
            # ถ้าเป็น 2D target
            elif len(self.target_shape) == 2:
                target_rows, target_cols = self.target_shape
                
                if len(current_shape) == 1:
                    # 1D -> 2D: reshape หรือ repeat
                    current_size = current_shape[0]
                    target_size = target_rows * target_cols
                    
                    if current_size >= target_size:
                        # ตัดแล้ว reshape
                        resized = obs[:target_size]
                        return resized.reshape(target_rows, target_cols)
                    else:
                        # เพิ่ม padding แล้ว reshape
                        padding = np.zeros(target_size - current_size)
                        padded = np.concatenate([obs, padding])
                        return padded.reshape(target_rows, target_cols)
                        
                elif len(current_shape) == 2:
                    # 2D -> 2D: resize rows และ columns
                    current_rows, current_cols = current_shape
                    
                    # ปรับ rows
                    if current_rows > target_rows:
                        obs = obs[:target_rows, :]
                    elif current_rows < target_rows:
                        padding_rows = np.zeros((target_rows - current_rows, current_cols))
                        obs = np.vstack([obs, padding_rows])
                    
                    # ปรับ columns
                    if current_cols > target_cols:
                        obs = obs[:, :target_cols]
                    elif current_cols < target_cols:
                        padding_cols = np.zeros((target_rows, target_cols - current_cols))
                        obs = np.hstack([obs, padding_cols])
                    
                    return obs
            
            # ถ้าไม่สามารถแปลงได้ ให้ใช้ zero array
            print(f"   ⚠️ Cannot convert {current_shape} to {self.target_shape}, using zeros")
            return np.zeros(self.target_shape, dtype=np.float32)
            
        except Exception as e:
            print(f"   ❌ Observation conversion error: {e}")
            # ส่งคืน zero array เป็น fallback
            return np.zeros(self.target_shape, dtype=np.float32)

class ModelBiasValidator:
    """
    🔍 Model Bias Validator - ตรวจสอบ bias ในโมเดลทุกตัวใน models/
    """
    
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.bias_models_dir = "biasmodel"  # Directory for biased models
        self.symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCAD', 'AUDUSD', 'XAUUSD']
        self.action_names = {0: 'HOLD', 1: 'BUY', 2: 'SELL', 3: 'CLOSE'}
        
        # Bias detection thresholds (same as training files)
        self.bias_thresholds = {
            'severe_hold_bias': 95.0,          # >95% hold actions
            'zero_trading_bias': 0,            # No trades at all
            'action_concentration_bias': 97.0, # >97% of one action type
            'poor_exploration_bias': 0.1       # <0.1% trading frequency
        }
        
        # Create biasmodel directory if it doesn't exist
        if not os.path.exists(self.bias_models_dir):
            os.makedirs(self.bias_models_dir)
            print(f"📁 Created bias models directory: {self.bias_models_dir}")
        
        print("🔍 MODEL BIAS VALIDATOR INITIALIZED")
        print(f"   📁 Models Directory: {self.models_dir}")
        print(f"   🗂️ Bias Models Directory: {self.bias_models_dir}")
        print(f"   📊 Symbols: {', '.join(self.symbols)}")
        print(f"   🎯 Bias Thresholds: Hold>{self.bias_thresholds['severe_hold_bias']}%, "
              f"Concentration>{self.bias_thresholds['action_concentration_bias']}%, "
              f"Trading<{self.bias_thresholds['poor_exploration_bias']}%")
    
    def find_all_models(self):
        """ค้นหาโมเดลทั้งหมดใน models/ directory"""
        all_models = []
        
        # ค้นหาในทุก tier directories
        tier_dirs = ['diamond', 'gold', 'silver', 'bronze', 'successful','active_trader']

        for tier in tier_dirs:
            tier_path = os.path.join(self.models_dir, tier)
            if os.path.exists(tier_path):
                # ค้นหา .zip files
                zip_files = glob.glob(os.path.join(tier_path, "*.zip"))
                
                for zip_file in zip_files:
                    try:
                        # Extract model info from filename
                        filename = os.path.basename(zip_file)
                        
                        # ดึง symbol จาก filename
                        symbol = None
                        for sym in self.symbols:
                            if sym.lower() in filename.lower():
                                symbol = sym
                                break
                        
                        if not symbol:
                            continue
                        
                        # ดึง score จาก filename
                        score = 0
                        if "_score" in filename:
                            try:
                                score_part = filename.split("_score")[1].split("_")[0]
                                score = float(score_part)
                            except:
                                score = 0
                        
                        # ดึง timestamp จาก filename
                        timestamp = "unknown"
                        import re
                        timestamp_match = re.search(r'(\d{8}_\d{6})', filename)
                        if timestamp_match:
                            timestamp = timestamp_match.group(1)
                        
                        all_models.append({
                            'path': zip_file,
                            'tier': tier,
                            'symbol': symbol,
                            'score': score,
                            'timestamp': timestamp,
                            'filename': filename
                        })
                        
                    except Exception as e:
                        print(f"⚠️ Error processing {zip_file}: {e}")
                        continue
        
        # เรียงตาม tier และ score
        tier_order = {'diamond': 4, 'gold': 3, 'silver': 2, 'bronze': 1, 'successful': 0, 'active_trader': 5}
        all_models.sort(key=lambda x: (tier_order.get(x['tier'], 0), x['score']), reverse=True)
        
        return all_models
    
    def load_model_safely(self, model_path):
        """โหลดโมเดลอย่างปลอดภัย ลองทุก algorithm"""
        algorithms = ['PPO', 'SAC', 'A2C']
        
        for alg in algorithms:
            try:
                if alg == 'PPO':
                    model = PPO.load(model_path)
                elif alg == 'SAC':
                    model = SAC.load(model_path)
                elif alg == 'A2C':
                    model = A2C.load(model_path)
                
                print(f"   ✅ Loaded as {alg}")
                return model, alg
                
            except Exception as e:
                print(f"   ⚠️ Failed as {alg}: {str(e)[:50]}...")
                continue
        
        return None, None
    
    def create_test_environment(self, symbol, model=None):
        """สร้าง test environment สำหรับ symbol ที่ตรงกับ observation shape ของโมเดล"""
        try:
            # ใช้ SimpleForexBot เพื่อสร้าง environment
            bot = SimpleForexBot(symbol, 'smart_defaults')
            
            # ถ้ามีโมเดล ให้ตรวจสอบ observation shape
            if model and hasattr(model, 'observation_space'):
                obs_shape = model.observation_space.shape
                print(f"   🔍 Model expects observation shape: {obs_shape}")
                
                # ปรับ lookback_window ตาม observation shape
                if len(obs_shape) == 2:  # 2D observation like (25, 13) or (50, 13)
                    expected_lookback = obs_shape[0]
                    print(f"   🔧 Adjusting lookback_window to {expected_lookback}")
                    bot.lookback_window = expected_lookback
                elif len(obs_shape) == 1:  # 1D observation
                    # สำหรับ 1D observation ให้ใช้ default lookback
                    expected_lookback = 50  # หรือค่าที่เหมาะสม
                    print(f"   🔧 Using default lookback_window {expected_lookback} for 1D observation")
                    bot.lookback_window = expected_lookback
            
            env = bot.create_environment()
            
            # ตรวจสอบ observation space ของ environment ที่สร้าง
            if hasattr(env, 'observation_space'):
                env_obs_shape = env.observation_space.shape
                print(f"   📐 Environment observation shape: {env_obs_shape}")
                
                # ถ้า shape ไม่ตรงกัน แจ้งเตือน
                if model and hasattr(model, 'observation_space'):
                    model_obs_shape = model.observation_space.shape
                    if env_obs_shape != model_obs_shape:
                        print(f"   ⚠️ Shape mismatch: Model={model_obs_shape}, Env={env_obs_shape}")
                        # พยายามปรับอีกครั้ง
                        return self.create_compatible_environment(symbol, model_obs_shape)
            
            print(f"   ✅ Created environment for {symbol}")
            return env
        except Exception as e:
            print(f"   ❌ Failed to create environment: {e}")
            return None
    
    def create_compatible_environment(self, symbol, target_shape):
        """สร้าง environment ที่เข้ากันได้กับ target observation shape"""
        try:
            print(f"   🔄 Creating compatible environment with shape {target_shape}")
            
            # สร้าง bot ใหม่พร้อมปรับ parameters
            bot = SimpleForexBot(symbol, 'smart_defaults')
            
            if len(target_shape) == 2:  # 2D observation
                lookback, features = target_shape
                bot.lookback_window = lookback
                print(f"   🎯 Set lookback_window to {lookback} for {features} features")
            elif len(target_shape) == 1:  # 1D observation
                bot.lookback_window = min(target_shape[0], 100)  # จำกัดไม่ให้ใหญ่เกินไป
                print(f"   🎯 Set lookback_window to {bot.lookback_window} for 1D observation")
            
            env = bot.create_environment()
            
            # ตรวจสอบอีกครั้ง
            if hasattr(env, 'observation_space'):
                actual_shape = env.observation_space.shape
                print(f"   📐 Final environment shape: {actual_shape}")
                
                if actual_shape != target_shape:
                    print(f"   ⚠️ Still mismatched - using wrapper approach")
                    # ใช้ wrapper เพื่อปรับ observation
                    env = ObservationShapeWrapper(env, target_shape)
                    print(f"   🔧 Applied observation wrapper")
            
            return env
            
        except Exception as e:
            print(f"   ❌ Failed to create compatible environment: {e}")
            return None
    
    def detect_model_bias(self, model, env, num_episodes=3, max_steps_per_episode=500):
        """ตรวจสอบ bias ในโมเดล - ใช้ logic เดียวกับ training files"""
        print(f"      🧪 Testing model behavior ({num_episodes} episodes, max {max_steps_per_episode} steps each)...")
        
        bias_flags = {
            'severe_hold_bias': False,
            'zero_trading_bias': False,
            'action_concentration_bias': False,
            'poor_exploration_bias': False
        }
        
        total_actions = {'hold': 0, 'buy': 0, 'sell': 0, 'close': 0}
        total_trades = 0
        total_steps = 0
        all_raw_actions = []
        
        try:
            for episode in range(num_episodes):
                try:
                    obs, _ = env.reset()
                except Exception as reset_error:
                    print(f"      ❌ Environment reset failed: {reset_error}")
                    return None, f"Environment reset failed: {reset_error}"
                    
                episode_steps = 0
                episode_trades = 0
                
                for step in range(max_steps_per_episode):
                    try:
                        # Get model prediction
                        action, _ = model.predict(obs, deterministic=True)
                    except Exception as predict_error:
                        print(f"      ❌ Model prediction failed: {predict_error}")
                        if "observation shape" in str(predict_error).lower():
                            return None, f"Observation shape mismatch: {predict_error}"
                        else:
                            return None, f"Model prediction failed: {predict_error}"
                    
                    # Convert continuous action to discrete (same logic as training)
                    if isinstance(action, (list, np.ndarray)):
                        raw_action = float(action[0])
                    else:
                        raw_action = float(action)
                    
                    all_raw_actions.append(raw_action)
                    
                    # Map to discrete action (exact same logic as training)
                    if raw_action < -0.3:
                        discrete_action = 2  # Sell
                        action_type = 'sell'
                    elif raw_action < 0.3:
                        discrete_action = 0  # Hold
                        action_type = 'hold'
                    elif raw_action < 0.7:
                        discrete_action = 1  # Buy
                        action_type = 'buy'
                    else:
                        discrete_action = 3  # Close
                        action_type = 'close'
                    
                    # Count actions
                    total_actions[action_type] += 1
                    
                    # Count trades (Buy/Sell actions)
                    if action_type in ['buy', 'sell']:
                        total_trades += 1
                        episode_trades += 1
                    
                    # Execute action in environment
                    try:
                        result = env.step(discrete_action)
                        
                        # Handle different return formats
                        if len(result) == 5:  # New format: obs, reward, done, truncated, info
                            obs, reward, done, truncated, info = result
                        elif len(result) == 4:  # Old format: obs, reward, done, info
                            obs, reward, done, info = result
                            truncated = False
                        else:
                            print(f"      ⚠️ Unexpected env.step() return format: {len(result)} items")
                            obs, reward, done = result[0], result[1], result[2]
                            truncated = False
                            info = {}
                            
                    except Exception as step_error:
                        print(f"      ❌ Environment step failed: {step_error}")
                        break
                    
                    episode_steps += 1
                    total_steps += 1
                    
                    if done or truncated:
                        break
                
                print(f"         Episode {episode+1}: {episode_steps} steps, {episode_trades} trades")
        
        except Exception as e:
            print(f"      ❌ Error during testing: {e}")
            return None, None
        
        # Calculate percentages
        if total_steps == 0:
            return None, "No steps completed"
        
        action_percentages = {}
        for action_type, count in total_actions.items():
            action_percentages[action_type] = (count / total_steps) * 100
        
        # Detect biases (exact same logic as training files)
        
        # 1. Severe Hold Bias: >95% hold actions
        if action_percentages['hold'] > self.bias_thresholds['severe_hold_bias']:
            bias_flags['severe_hold_bias'] = True
        
        # 2. Zero Trading Bias: No trades at all
        if total_trades == 0:
            bias_flags['zero_trading_bias'] = True
        
        # 3. Action Concentration Bias: >97% of one type of action
        max_action_pct = max(action_percentages.values())
        if max_action_pct > self.bias_thresholds['action_concentration_bias']:
            bias_flags['action_concentration_bias'] = True
        
        # 4. Poor Exploration Bias: Trading frequency < 0.1%
        trading_frequency = total_trades / max(total_steps, 1)
        if trading_frequency < (self.bias_thresholds['poor_exploration_bias'] / 100):
            bias_flags['poor_exploration_bias'] = True
        
        # Calculate overall bias score
        severe_bias_count = sum(bias_flags.values())
        has_severe_bias = severe_bias_count >= 3  # 3 or more severe biases
        
        # Additional analysis
        raw_variance = np.var(all_raw_actions) if all_raw_actions else 0
        raw_mean = np.mean(all_raw_actions) if all_raw_actions else 0
        
        bias_report = {
            'total_steps': total_steps,
            'total_trades': total_trades,
            'trading_frequency': trading_frequency * 100,
            'action_percentages': action_percentages,
            'bias_flags': bias_flags,
            'severe_bias_count': severe_bias_count,
            'has_severe_bias': has_severe_bias,
            'raw_action_variance': raw_variance,
            'raw_action_mean': raw_mean,
            'verdict': 'BIASED' if has_severe_bias else 'ACCEPTABLE'
        }
        
        return bias_report, None
    
    def move_biased_model(self, model_info, bias_report):
        """ย้ายโมเดลที่มี bias ไปยังโฟลเดอร์ biasmodel"""
        try:
            source_path = model_info['path']
            filename = model_info['filename']
            
            # สร้าง subdirectory ใน biasmodel ตาม tier เดิม
            bias_tier_dir = os.path.join(self.bias_models_dir, model_info['tier'])
            if not os.path.exists(bias_tier_dir):
                os.makedirs(bias_tier_dir)
            
            # เพิ่มข้อมูล bias ลงในชื่อไฟล์
            base_name, ext = os.path.splitext(filename)
            bias_flags = bias_report['bias_flags']
            bias_types = [bias_type for bias_type, detected in bias_flags.items() if detected]
            bias_suffix = "_BIAS_" + "_".join([bt.split('_')[0].upper() for bt in bias_types])
            
            new_filename = f"{base_name}{bias_suffix}{ext}"
            destination_path = os.path.join(bias_tier_dir, new_filename)
            
            # ย้ายไฟล์
            shutil.move(source_path, destination_path)
            
            # สร้างไฟล์ report สำหรับโมเดลที่ย้าย
            report_file = os.path.join(bias_tier_dir, f"{base_name}_bias_report.json")
            report_data = {
                'original_path': source_path,
                'moved_to': destination_path,
                'moved_at': datetime.now().isoformat(),
                'bias_report': bias_report,
                'model_info': model_info,
                'reason': 'Model detected with severe bias'
            }
            
            with open(report_file, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            print(f"      🗂️ MOVED TO BIAS FOLDER:")
            print(f"         From: {source_path}")
            print(f"         To: {destination_path}")
            print(f"         Report: {report_file}")
            
            return True, destination_path
            
        except Exception as e:
            print(f"      ❌ Failed to move model: {e}")
            return False, str(e)
    
    def restore_model_from_bias(self, biased_model_path, original_tier=None):
        """คืนโมเดลจากโฟลเดอร์ bias กลับไปที่เดิม (ถ้าจำเป็น)"""
        try:
            # ดึงข้อมูลจาก bias report
            model_dir = os.path.dirname(biased_model_path)
            filename = os.path.basename(biased_model_path)
            base_name = os.path.splitext(filename)[0]
            
            # หา bias report file
            report_file = os.path.join(model_dir, f"{base_name.split('_BIAS_')[0]}_bias_report.json")
            
            if not os.path.exists(report_file):
                print(f"❌ Cannot find bias report file: {report_file}")
                return False, "Bias report not found"
            
            # อ่านข้อมูลจาก report
            with open(report_file, 'r') as f:
                report_data = json.load(f)
            
            original_path = report_data['original_path']
            original_dir = os.path.dirname(original_path)
            
            # สร้างโฟลเดอร์เดิมถ้าไม่มี
            if not os.path.exists(original_dir):
                os.makedirs(original_dir)
            
            # ย้ายกลับ (เอาส่วน bias ออกจากชื่อ)
            original_filename = os.path.basename(original_path)
            restored_path = os.path.join(original_dir, original_filename)
            
            shutil.move(biased_model_path, restored_path)
            
            # ลบ bias report
            os.remove(report_file)
            
            print(f"✅ Model restored:")
            print(f"   From: {biased_model_path}")
            print(f"   To: {restored_path}")
            
            return True, restored_path
            
        except Exception as e:
            print(f"❌ Failed to restore model: {e}")
            return False, str(e)
    
    def list_biased_models(self):
        """แสดงรายการโมเดลที่ถูกย้ายไปยัง bias folder"""
        print(f"\n🗂️ BIASED MODELS DIRECTORY: {self.bias_models_dir}")
        print("=" * 50)
        
        if not os.path.exists(self.bias_models_dir):
            print("❌ No biased models directory found")
            return []
        
        biased_models = []
        
        # ค้นหาใน tier directories
        for tier in ['diamond', 'gold', 'silver', 'bronze', 'successful']:
            tier_path = os.path.join(self.bias_models_dir, tier)
            if os.path.exists(tier_path):
                models = glob.glob(os.path.join(tier_path, "*.zip"))
                reports = glob.glob(os.path.join(tier_path, "*_bias_report.json"))
                
                print(f"\n📁 {tier.upper()} TIER:")
                print(f"   Models: {len(models)}")
                print(f"   Reports: {len(reports)}")
                
                for model_path in models:
                    filename = os.path.basename(model_path)
                    
                    # หา bias report ที่เกี่ยวข้อง
                    base_name = filename.split('_BIAS_')[0] if '_BIAS_' in filename else os.path.splitext(filename)[0]
                    report_path = os.path.join(tier_path, f"{base_name}_bias_report.json")
                    
                    bias_info = None
                    if os.path.exists(report_path):
                        try:
                            with open(report_path, 'r') as f:
                                bias_info = json.load(f)
                        except:
                            pass
                    
                    biased_models.append({
                        'path': model_path,
                        'tier': tier,
                        'filename': filename,
                        'report_path': report_path if os.path.exists(report_path) else None,
                        'bias_info': bias_info
                    })
                    
                    # แสดงข้อมูลโมเดล
                    if bias_info and 'bias_report' in bias_info:
                        br = bias_info['bias_report']
                        bias_flags = [bt for bt, detected in br['bias_flags'].items() if detected]
                        moved_date = bias_info.get('moved_at', 'Unknown')[:10]  # แค่วันที่
                        print(f"      📄 {filename[:40]}...")
                        print(f"         Bias Types: {', '.join([bf.split('_')[0].upper() for bf in bias_flags])}")
                        print(f"         Moved: {moved_date}")
                    else:
                        print(f"      📄 {filename} (no report)")
        
        print(f"\n📊 TOTAL BIASED MODELS: {len(biased_models)}")
        return biased_models
    
    def cleanup_biased_models(self, confirm=True):
        """ลบโมเดลที่มี bias ทั้งหมด (ระวัง!)"""
        biased_models = self.list_biased_models()
        
        if not biased_models:
            print("✅ No biased models to cleanup")
            return
        
        if confirm:
            print(f"\n⚠️ WARNING: This will PERMANENTLY DELETE {len(biased_models)} biased models!")
            confirm_input = input("Type 'DELETE' to confirm: ").strip()
            if confirm_input != 'DELETE':
                print("❌ Cleanup cancelled")
                return
        
        deleted_count = 0
        for model in biased_models:
            try:
                # ลบไฟล์โมเดล
                if os.path.exists(model['path']):
                    os.remove(model['path'])
                    deleted_count += 1
                    print(f"🗑️ Deleted: {model['filename']}")
                
                # ลบ bias report
                if model['report_path'] and os.path.exists(model['report_path']):
                    os.remove(model['report_path'])
                    print(f"📄 Deleted report: {os.path.basename(model['report_path'])}")
                    
            except Exception as e:
                print(f"❌ Failed to delete {model['filename']}: {e}")
        
        print(f"\n✅ Cleanup complete: {deleted_count}/{len(biased_models)} models deleted")
        
        # ลบโฟลเดอร์ว่างเปล่า
        try:
            for tier in ['diamond', 'gold', 'silver', 'bronze', 'successful']:
                tier_path = os.path.join(self.bias_models_dir, tier)
                if os.path.exists(tier_path) and not os.listdir(tier_path):
                    os.rmdir(tier_path)
                    print(f"📁 Removed empty directory: {tier}")
        except:
            pass
    
    def get_bias_statistics(self):
        """ได้สถิติโดยรวมของ bias ใน models และ biasmodel directories"""
        print("\n📊 COMPREHENSIVE BIAS STATISTICS")
        print("=" * 50)
        
        # นับโมเดลใน models directory
        active_models = self.find_all_models()
        active_count = len(active_models)
        
        # นับโมเดลใน biasmodel directory  
        biased_models = []
        if os.path.exists(self.bias_models_dir):
            biased_models = self.list_biased_models()
        biased_count = len(biased_models)
        
        total_models = active_count + biased_count
        
        print(f"📁 ACTIVE MODELS (in {self.models_dir}/): {active_count}")
        print(f"🗂️ BIASED MODELS (in {self.bias_models_dir}/): {biased_count}")
        print(f"📊 TOTAL MODELS: {total_models}")
        
        if total_models > 0:
            bias_rate = (biased_count / total_models) * 100
            print(f"📈 OVERALL BIAS RATE: {bias_rate:.1f}%")
            
            if bias_rate > 50:
                print("🚨 HIGH BIAS RATE - Review training process!")
            elif bias_rate > 20:
                print("⚠️ MODERATE BIAS RATE - Some improvements needed")
            else:
                print("✅ LOW BIAS RATE - Good training quality")
        
        # แยกตาม symbol
        symbol_stats = {}
        
        # Active models by symbol
        for model in active_models:
            symbol = model['symbol']
            if symbol not in symbol_stats:
                symbol_stats[symbol] = {'active': 0, 'biased': 0}
            symbol_stats[symbol]['active'] += 1
        
        # Biased models by symbol
        for model in biased_models:
            if model['bias_info'] and 'model_info' in model['bias_info']:
                symbol = model['bias_info']['model_info'].get('symbol', 'Unknown')
                if symbol != 'Unknown':
                    if symbol not in symbol_stats:
                        symbol_stats[symbol] = {'active': 0, 'biased': 0}
                    symbol_stats[symbol]['biased'] += 1
        
        print(f"\n💱 BY SYMBOL:")
        for symbol in sorted(symbol_stats.keys()):
            stats = symbol_stats[symbol]
            total_sym = stats['active'] + stats['biased']
            bias_rate_sym = (stats['biased'] / total_sym * 100) if total_sym > 0 else 0
            print(f"   {symbol}: {stats['active']} active, {stats['biased']} biased ({bias_rate_sym:.1f}% bias rate)")
        
        return {
            'active_models': active_count,
            'biased_models': biased_count,
            'total_models': total_models,
            'overall_bias_rate': (biased_count / total_models * 100) if total_models > 0 else 0,
            'symbol_stats': symbol_stats
        }
    
    def validate_single_model(self, model_info):
        """ตรวจสอบโมเดลหนึ่งตัว"""
        print(f"\n🔍 VALIDATING: {model_info['filename']}")
        print(f"   📊 Tier: {model_info['tier'].upper()}")
        print(f"   💎 Symbol: {model_info['symbol']}")
        print(f"   📈 Score: {model_info['score']:.1f}")
        print(f"   📅 Timestamp: {model_info['timestamp']}")
        
        # Load model
        print(f"   🔄 Loading model...")
        model, algorithm = self.load_model_safely(model_info['path'])
        
        if not model:
            return {
                'model_info': model_info,
                'status': 'LOAD_FAILED',
                'error': 'Failed to load model',
                'bias_report': None
            }
        
        # Create environment
        print(f"   🏗️ Setting up test environment...")
        env = self.create_test_environment(model_info['symbol'], model)
        
        if not env:
            return {
                'model_info': model_info,
                'status': 'ENV_FAILED',
                'error': 'Failed to create environment',
                'bias_report': None
            }
        
        # Test for bias
        bias_report, error = self.detect_model_bias(model, env)
        
        if error:
            return {
                'model_info': model_info,
                'status': 'TEST_FAILED',
                'error': error,
                'bias_report': None
            }
        
        # Print detailed results
        print(f"      📊 BIAS ANALYSIS RESULTS:")
        print(f"         Total Steps: {bias_report['total_steps']}")
        print(f"         Total Trades: {bias_report['total_trades']}")
        print(f"         Trading Frequency: {bias_report['trading_frequency']:.2f}%")
        print(f"         Raw Action Variance: {bias_report['raw_action_variance']:.6f}")
        print(f"         Raw Action Mean: {bias_report['raw_action_mean']:.3f}")
        
        print(f"      📋 ACTION DISTRIBUTION:")
        for action, pct in bias_report['action_percentages'].items():
            status = "🚨" if pct > 80 else "⚠️" if pct > 60 else "✅"
            print(f"         {status} {action.upper()}: {pct:.1f}%")
        
        print(f"      🚨 BIAS FLAGS:")
        bias_detected = 0
        for bias_type, detected in bias_report['bias_flags'].items():
            status = "🚨 DETECTED" if detected else "✅ OK"
            if detected:
                bias_detected += 1
            print(f"         {bias_type}: {status}")
        
        verdict = "🚨 BIASED" if bias_report['has_severe_bias'] else "✅ ACCEPTABLE"
        print(f"      🏁 VERDICT: {verdict} ({bias_detected}/4 bias flags)")
        
        # ย้ายโมเดลที่มี bias ไปยังโฟลเดอร์ biasmodel
        moved_to_bias = False
        if bias_report['has_severe_bias']:
            print(f"      🗂️ Moving biased model to bias folder...")
            success, new_path = self.move_biased_model(model_info, bias_report)
            if success:
                moved_to_bias = True
                model_info['moved_to_bias'] = True
                model_info['bias_path'] = new_path
            else:
                print(f"      ⚠️ Failed to move model: {new_path}")
        
        return {
            'model_info': model_info,
            'algorithm': algorithm,
            'status': 'COMPLETED',
            'bias_report': bias_report,
            'moved_to_bias': moved_to_bias,
            'error': None
        }
    
    def validate_all_models(self, max_models=None):
        """ตรวจสอบโมเดลทั้งหมด"""
        print("🔍 STARTING COMPREHENSIVE MODEL BIAS VALIDATION")
        print("=" * 60)
        
        # Find all models
        all_models = self.find_all_models()
        
        if not all_models:
            print("❌ No models found in models/ directory")
            return []
        
        print(f"📋 Found {len(all_models)} models to validate")
        
        if max_models:
            all_models = all_models[:max_models]
            print(f"🎯 Limiting to first {max_models} models")
        
        # Validate each model
        results = []
        
        for i, model_info in enumerate(all_models, 1):
            print(f"\n{'='*60}")
            print(f"📝 PROGRESS: {i}/{len(all_models)}")
            
            try:
                result = self.validate_single_model(model_info)
                results.append(result)
                
                # Short summary
                if result['status'] == 'COMPLETED' and result['bias_report']:
                    verdict = result['bias_report']['verdict']
                    bias_count = result['bias_report']['severe_bias_count']
                    print(f"   ✅ Completed: {verdict} ({bias_count}/4 bias flags)")
                else:
                    print(f"   ❌ Failed: {result['error']}")
                    
            except Exception as e:
                print(f"   💥 Exception: {e}")
                results.append({
                    'model_info': model_info,
                    'status': 'EXCEPTION',
                    'error': str(e),
                    'bias_report': None
                })
            
            # Add small delay to prevent system overload
            import time
            time.sleep(1)
        
        return results
    
    def generate_summary_report(self, results):
        """สร้างรายงานสรุป"""
        print(f"\n{'='*60}")
        print("📊 VALIDATION SUMMARY REPORT")
        print("=" * 60)
        
        if not results:
            print("❌ No results to report")
            return
        
        # Count by status
        status_counts = {}
        for result in results:
            status = result['status']
            status_counts[status] = status_counts.get(status, 0) + 1
        
        print(f"📈 VALIDATION STATUS:")
        for status, count in status_counts.items():
            print(f"   {status}: {count}")
        
        # Analyze completed validations
        completed_results = [r for r in results if r['status'] == 'COMPLETED' and r['bias_report']]
        
        if not completed_results:
            print("❌ No successful validations to analyze")
            return
        
        print(f"\n📊 BIAS ANALYSIS ({len(completed_results)} models):")
        
        # Count by verdict
        biased_count = sum(1 for r in completed_results if r['bias_report']['has_severe_bias'])
        acceptable_count = len(completed_results) - biased_count
        moved_count = sum(1 for r in completed_results if r.get('moved_to_bias', False))
        
        print(f"   ✅ ACCEPTABLE: {acceptable_count}")
        print(f"   🚨 BIASED: {biased_count}")
        print(f"   🗂️ MOVED TO BIAS FOLDER: {moved_count}")
        print(f"   📊 Success Rate: {(acceptable_count/len(completed_results)*100):.1f}%")
        
        # Bias flag statistics
        bias_flag_counts = {}
        for result in completed_results:
            for bias_type, detected in result['bias_report']['bias_flags'].items():
                if detected:
                    bias_flag_counts[bias_type] = bias_flag_counts.get(bias_type, 0) + 1
        
        print(f"\n🚨 MOST COMMON BIAS TYPES:")
        for bias_type, count in sorted(bias_flag_counts.items(), key=lambda x: x[1], reverse=True):
            pct = (count / len(completed_results)) * 100
            print(f"   {bias_type}: {count}/{len(completed_results)} ({pct:.1f}%)")
        
        # Tier analysis
        tier_stats = {}
        for result in completed_results:
            tier = result['model_info']['tier']
            if tier not in tier_stats:
                tier_stats[tier] = {'total': 0, 'biased': 0}
            tier_stats[tier]['total'] += 1
            if result['bias_report']['has_severe_bias']:
                tier_stats[tier]['biased'] += 1
        
        print(f"\n🏆 BIAS BY TIER:")
        for tier in ['diamond', 'gold', 'silver', 'bronze', 'successful']:
            if tier in tier_stats:
                stats = tier_stats[tier]
                bias_rate = (stats['biased'] / stats['total']) * 100
                print(f"   {tier.upper()}: {stats['biased']}/{stats['total']} biased ({bias_rate:.1f}%)")
        
        # Symbol analysis
        symbol_stats = {}
        for result in completed_results:
            symbol = result['model_info']['symbol']
            if symbol not in symbol_stats:
                symbol_stats[symbol] = {'total': 0, 'biased': 0}
            symbol_stats[symbol]['total'] += 1
            if result['bias_report']['has_severe_bias']:
                symbol_stats[symbol]['biased'] += 1
        
        print(f"\n💱 BIAS BY SYMBOL:")
        for symbol in sorted(symbol_stats.keys()):
            stats = symbol_stats[symbol]
            bias_rate = (stats['biased'] / stats['total']) * 100
            print(f"   {symbol}: {stats['biased']}/{stats['total']} biased ({bias_rate:.1f}%)")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if biased_count > acceptable_count:
            print("   🚨 HIGH BIAS RATE - Most models are biased!")
            print("   📋 Actions needed:")
            print("      1. Review training hyperparameters")
            print("      2. Improve training data diversity")
            print("      3. Add regularization techniques")
            print("      4. Extend training time")
        elif biased_count > 0:
            print("   ⚠️ Some models show bias - selective retraining needed")
            print("   📋 Focus on retraining biased models only")
        else:
            print("   ✅ All models look healthy - good job!")
        
        if moved_count > 0:
            print(f"\n🗂️ BIASED MODELS MANAGEMENT:")
            print(f"   📁 {moved_count} biased models moved to '{self.bias_models_dir}' folder")
            print(f"   💡 You can:")
            print(f"      • Review them later: validator.list_biased_models()")
            print(f"      • Delete them permanently: validator.cleanup_biased_models()")
            print(f"      • Restore specific models: validator.restore_model_from_bias(path)")
        
        # Save report to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"model_bias_validation_report_{timestamp}.json"
        
        report_data = {
            'timestamp': timestamp,
            'total_models': len(results),
            'completed_validations': len(completed_results),
            'biased_count': biased_count,
            'acceptable_count': acceptable_count,
            'success_rate': (acceptable_count/len(completed_results)*100) if completed_results else 0,
            'bias_flag_counts': bias_flag_counts,
            'tier_stats': tier_stats,
            'symbol_stats': symbol_stats,
            'detailed_results': results
        }
        
        try:
            with open(report_file, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            print(f"\n📁 Detailed report saved to: {report_file}")
        except Exception as e:
            print(f"⚠️ Failed to save report: {e}")

def main():
    """Main validation function"""
    print("🔍 MODEL BIAS VALIDATOR")
    print("=" * 40)
    
    try:
        validator = ModelBiasValidator()
        
        while True:
            print("\n📋 VALIDATION OPTIONS:")
            print("1. Validate ALL models (may take long time)")
            print("2. Validate top 10 models only")
            print("3. Validate top 5 models only")
            print("4. Quick test (top 3 models)")
            print("5. 🗂️ List biased models")
            print("6. 🧹 Cleanup biased models (DELETE)")
            print("7. 🔄 Restore model from bias folder")
            print("8. 📊 Show bias statistics")
            print("0. ❌ Exit")
            
            choice = input("👉 Select option (0-8): ").strip()
            
            if choice == '0':
                print("👋 Goodbye!")
                break
            elif choice == '5':
                # List biased models
                validator.list_biased_models()
                input("\n👉 Press Enter to continue...")
                continue
            elif choice == '6':
                # Cleanup biased models
                validator.cleanup_biased_models()
                input("\n👉 Press Enter to continue...")
                continue
            elif choice == '7':
                # Restore model from bias folder
                biased_models = validator.list_biased_models()
                if not biased_models:
                    print("❌ No biased models found to restore")
                    input("\n👉 Press Enter to continue...")
                    continue
                
                print(f"\n📋 SELECT MODEL TO RESTORE:")
                for i, model in enumerate(biased_models, 1):
                    print(f"{i}. {model['filename'][:50]}... ({model['tier']})")
                
                try:
                    restore_choice = int(input("👉 Select model number: ").strip()) - 1
                    if 0 <= restore_choice < len(biased_models):
                        selected_model = biased_models[restore_choice]
                        print(f"🔄 Restoring: {selected_model['filename']}")
                        success, result = validator.restore_model_from_bias(selected_model['path'])
                        if success:
                            print(f"✅ Model restored successfully to: {result}")
                        else:
                            print(f"❌ Failed to restore: {result}")
                    else:
                        print("❌ Invalid selection")
                except (ValueError, IndexError):
                    print("❌ Invalid input")
                
                input("\n👉 Press Enter to continue...")
                continue
            elif choice == '8':
                # Show bias statistics
                validator.get_bias_statistics()
                input("\n👉 Press Enter to continue...")
                continue
            
            # Validation options
            max_models = None
            if choice == '2':
                max_models = 10
            elif choice == '3':
                max_models = 5
            elif choice == '4':
                max_models = 3
            elif choice != '1':
                print("❌ Invalid choice")
                continue
            
            # Run validation
            print(f"\n🚀 Starting validation...")
            results = validator.validate_all_models(max_models)
            
            # Generate summary report
            validator.generate_summary_report(results)
            
            input("\n👉 Press Enter to continue...")
        
    except KeyboardInterrupt:
        print("\n🛑 Validation stopped by user")
    except Exception as e:
        print(f"\n💥 Validation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
