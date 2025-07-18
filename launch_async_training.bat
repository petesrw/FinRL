@echo off
echo 🚀 Async Training System Launcher
echo ================================

echo.
echo 📊 Checking Python environment...
python --version
if errorlevel 1 (
    echo ❌ Python not found! Please install Python first.
    pause
    exit /b 1
)

echo.
echo 🔧 Checking required packages...
python -c "import torch; print(f'✅ PyTorch: {torch.__version__}')" 2>nul
if errorlevel 1 (
    echo ❌ PyTorch not found! Installing...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
)

python -c "import stable_baselines3; print(f'✅ Stable-Baselines3: {stable_baselines3.__version__}')" 2>nul
if errorlevel 1 (
    echo ❌ Stable-Baselines3 not found! Installing...
    pip install stable-baselines3
)

python -c "import pandas; print(f'✅ Pandas: {pandas.__version__}')" 2>nul
if errorlevel 1 (
    echo ❌ Pandas not found! Installing...
    pip install pandas
)

python -c "import gymnasium; print(f'✅ Gymnasium: {gymnasium.__version__}')" 2>nul
if errorlevel 1 (
    echo ❌ Gymnasium not found! Installing...
    pip install gymnasium
)

echo.
echo 🎯 Select what to run:
echo.
echo 1. 🧪 Test Async System (Quick test)
echo 2. 🚀 Full Async Training (Production)
echo 3. 📊 System Information Only
echo 4. 📋 View README

set /p choice="Choose option (1-4): "

if "%choice%"=="1" (
    echo.
    echo 🧪 Running Async System Test...
    python test_async_training.py
) else if "%choice%"=="2" (
    echo.
    echo 🚀 Starting Full Async Training...
    python train_all_models.py
) else if "%choice%"=="3" (
    echo.
    echo 📊 System Information:
    python async_config.py
) else if "%choice%"=="4" (
    echo.
    echo 📋 Opening README...
    start ASYNC_TRAINING_README.md
) else (
    echo Invalid choice. Defaulting to test mode...
    echo.
    echo 🧪 Running Async System Test...
    python test_async_training.py
)

echo.
echo 🏁 Task completed!
pause
