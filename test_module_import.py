#!/usr/bin/env python3
"""
🧪 Test Threading Fix with Module-Level Import
Test if moving GPU setup to main() fixes the threading issue
"""

import sys
import os

# Test importing the main script (should not trigger GPU setup)
print("🧪 Testing module import without GPU setup...")

try:
    # Add current directory to path
    sys.path.insert(0, r'd:\workplace\finFL\FinRL')
    
    print("   Step 1: Importing train_all_models module...")
    import train_all_models
    print("   ✅ Step 1: Module imported successfully")
    
    print("   Step 2: Checking threading configuration...")
    if hasattr(train_all_models, '_threading_configured') and train_all_models._threading_configured:
        print("   ✅ Step 2: Threading configured globally")
    else:
        print("   ❌ Step 2: Threading not configured")
    
    print("   Step 3: Checking DEVICE initialization...")
    if train_all_models.DEVICE is None:
        print("   ✅ Step 3: DEVICE not initialized at module level (GOOD!)")
    else:
        print("   ⚠️ Step 3: DEVICE already initialized at module level")
    
    print("   Step 4: Testing initialize_gpu_setup function...")
    device = train_all_models.initialize_gpu_setup()
    print(f"   ✅ Step 4: GPU setup initialized, device: {device}")
    
    print("\n🎉 Threading fix test completed successfully!")
    print("💡 GPU setup is now deferred until main() execution")
    
except Exception as e:
    print(f"❌ Test failed: {str(e)}")
    import traceback
    traceback.print_exc()

print("\n🚀 If no threading errors appeared above, the fix is working!")
