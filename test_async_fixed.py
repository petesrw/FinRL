#!/usr/bin/env python3
"""
Quick test of async training with fixed SAC issue
"""

import asyncio
import sys
import os

# Add current directory to path
sys.path.append('.')

try:
    print("🧪 ทดสอบ async training ด้วยเกณฑ์ใหม่")
    print("="*50)
    
    # Test imports
    from train_all_models import AdaptiveTrainer
    import pandas as pd
    import numpy as np
    print("✅ Imports สำเร็จ")
    
    # Create test data
    np.random.seed(42)
    data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=2000, freq='H'),
        'open': np.random.randn(2000).cumsum() + 2000,
        'high': np.random.randn(2000).cumsum() + 2005,
        'low': np.random.randn(2000).cumsum() + 1995,
        'close': np.random.randn(2000).cumsum() + 2000
    })
    print("✅ ข้อมูลทดสอบพร้อม")
    
    # Test trainer creation
    trainer = AdaptiveTrainer('XAUUSD')
    print("✅ Trainer สร้างสำเร็จ")
    print(f"📊 เกณฑ์ Tier: {list(trainer.targets.keys())}")
    print(f"🔧 Max Concurrent: {trainer.max_concurrent_models}")
    print()
    
    # Start small async training test
    print("🚀 เริ่มการเทรน async ทดสอบ (4 attempts, batch_size=2)")
    result = asyncio.run(trainer.adaptive_train_async(
        data, 
        max_attempts=4, 
        target_tier='bronze', 
        batch_size=2
    ))
    
    if result:
        print("\n🎉 การเทรนเสร็จสิ้น!")
        print(f"🏆 Best Result: {result['tier']} - Score: {result['score']:.1f}")
        print(f"📈 Win Rate: {result['metrics']['win_rate']:.1%}")
        print(f"💰 Profit Factor: {result['metrics']['profit_factor']:.2f}")
        print(f"📉 Max Drawdown: {result['metrics']['max_drawdown']:.1%}")
    else:
        print("\n❌ ไม่พบผลลัพธ์ที่ดี แต่ระบบทำงานได้ปกติ")
    
    print("\n✅ ทดสอบเสร็จสิ้น!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
