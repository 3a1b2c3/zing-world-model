@echo off
setlocal enabledelayedexpansion

echo ==========================================
echo Zing World Model - Download Models
echo ==========================================
echo.

cd /d "%~dp0"

if not defined HF_TOKEN (
    echo ERROR: HF_TOKEN not set
    echo Set: set HF_TOKEN=your_huggingface_token
    exit /b 1
)

set MODELS_DIR=%cd%\pretrained_models
if not exist "%MODELS_DIR%" mkdir "%MODELS_DIR%"

echo Downloading models to: %MODELS_DIR%
echo.

echo [1/2] Downloading zing-0.5 base model...
call .venv\Scripts\activate.bat

python "%~dp0_download_models_helper.py" "%MODELS_DIR%"

if errorlevel 1 (
    echo WARNING: Model download failed
    echo Check HF_TOKEN and repo availability
)

echo.
echo [2/2] Setup complete!
echo.
echo Models directory: %MODELS_DIR%
echo Next: run_example.bat
echo.
