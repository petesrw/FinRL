2025-07-29 00:56:08,929 - forex_system_with_config - ERROR - Failed to load model: No module named 'numpy._core'
   ❌ Failed to load DIAMOND model

cd myenv\Lib\site-packages
mkdir numpy\_core

echo from numpy.core import * > numpy\_core\__init__.py