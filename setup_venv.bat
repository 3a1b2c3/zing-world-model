@echo off
setlocal enabledelayedexpansion

echo ==========================================
echo Zing World Model - Setup
echo ==========================================
echo.

cd /d "%~dp0"

echo [1/3] Creating Python venv...
py -3.11 -m venv .venv 2>nul
if errorlevel 1 (
    python3 -m venv .venv 2>nul
    if errorlevel 1 (
        python -m venv .venv
        if errorlevel 1 (
            echo ERROR: Python not found or too old
            echo Install Python 3.11+
            exit /b 1
        )
    )
)

echo.
echo [2/3] Installing dependencies...
call .venv\Scripts\activate.bat
if errorlevel 1 exit /b 1

pip install --upgrade pip setuptools wheel
if errorlevel 1 exit /b 1

echo Installing torch with CUDA 12.1...
pip install torch==2.9.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 exit /b 1

echo Installing other dependencies...
pip install --no-build-isolation -r requirements.txt
if errorlevel 1 (
    echo WARNING: Some packages failed to install
    echo Trying wheels-only install...
    pip install --only-binary :all: -r requirements.txt
    if errorlevel 1 exit /b 1
)

echo.
echo [3/3] Setup complete!
echo.
echo Next steps:
echo 1. Activate venv: .venv\Scripts\activate.bat
echo 2. Download models: download_models.bat
echo 3. Run example: run_example.bat
echo.
