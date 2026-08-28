@echo off
setlocal enabledelayedexpansion

echo ==========================================
echo Zing-0.5 on RTX 6000 (48GB)
echo ==========================================
echo.

cd /d "%~dp0"

where wsl >nul 2>&1
if errorlevel 1 (
    echo ERROR: WSL2 not found
    echo Install WSL2: wsl --install
    exit /b 1
)

echo Checking WSL2 CUDA...
wsl nvcc --version >nul 2>&1
if errorlevel 1 (
    echo WARNING: CUDA not detected in WSL2
    echo Install CUDA in WSL2: wsl sudo apt install nvidia-cuda-toolkit
    echo.
)

set WSL_HOME=%USERPROFILE%\zing-world-model
set WINDOWS_PROJECT=%cd%

echo Project: %WINDOWS_PROJECT%
echo WSL home: %WSL_HOME%
echo.

echo [1/3] Syncing project to WSL2...
wsl mkdir -p %WSL_HOME%
wsl cp -r /mnt/c/workspace/world/zing-world-model/* %WSL_HOME%/ 2>nul

echo [2/3] Setting up WSL2 venv...
wsl bash -c "cd %WSL_HOME% && python3.11 -m venv .venv 2>/dev/null || python3 -m venv .venv"
wsl bash -c "cd %WSL_HOME% && source .venv/bin/activate && pip install --quiet --upgrade pip setuptools wheel"
wsl bash -c "cd %WSL_HOME% && source .venv/bin/activate && pip install --quiet -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
wsl bash -c "cd %WSL_HOME% && source .venv/bin/activate && pip install --quiet -q -r requirements.txt"

echo [3/3] Running inference...
echo.

wsl bash -c "cd %WSL_HOME% && source .venv/bin/activate && bash run_rtx6000.sh"

if errorlevel 1 (
    echo.
    echo ERROR: Inference failed
    exit /b 1
)

echo.
echo Syncing outputs back to Windows...
xcopy /S /I /Y "%WSL_HOME%\outputs" "%WINDOWS_PROJECT%\outputs" >nul 2>&1

echo.
echo Complete! Output saved to: %WINDOWS_PROJECT%\outputs
echo.
pause
