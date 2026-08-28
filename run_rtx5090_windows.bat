@echo off
setlocal enabledelayedexpansion

echo ==========================================
echo Zing-0.5 on RTX 5090 (32GB)
echo ==========================================
echo.

cd /d "%~dp0"

REM Check if .venv exists
if not exist ".venv\Scripts\activate.bat" (
    echo ERROR: .venv not found
    echo Run: setup_venv.bat
    exit /b 1
)

REM Activate venv
call .venv\Scripts\activate.bat

REM Set environment variables
set PYTHONPATH=%cd%\src;%PYTHONPATH%
set CUDA_VISIBLE_DEVICES=0

set PRETRAINED_DIR=%cd%\pretrained_models
set OUTPUT_DIR=%cd%\outputs
set SNAPSHOTS_DIR=%PRETRAINED_DIR%\models--seedleap--zing-0.5\snapshots

if not exist "%PRETRAINED_DIR%" (
    echo ERROR: pretrained_models/ not found
    exit /b 1
)

REM Find the latest snapshot (first by directory sort)
for /d %%D in ("%SNAPSHOTS_DIR%\*") do (
    set SNAPSHOT_DIR=%%D
)

if not exist "%SNAPSHOT_DIR%\pretrained\text_encoder" (
    echo ERROR: Could not find Zing-0.5 models
    exit /b 1
)

set PRETRAINED_PATH=%SNAPSHOT_DIR%\pretrained
set CHECKPOINT_PATH=%SNAPSHOT_DIR%\generator\model.pt

echo Snapshot: %SNAPSHOT_DIR%
echo Output: %OUTPUT_DIR%\case3_rtx5090
echo.

echo Running case3 (long action T2V)...
echo.

python -m zing_v0_5 ^
  --pretrained-dir "%PRETRAINED_PATH%" ^
  --checkpoint "%CHECKPOINT_PATH%" ^
  --messages "examples\case3_action_t2v.jsonl" ^
  --output-dir "%OUTPUT_DIR%\case3_rtx5090" ^
  --local-attn-size 33 ^
  --sink-size 5 ^
  --seed 0

if errorlevel 1 (
    echo.
    echo ERROR: Inference failed
    exit /b 1
)

echo.
echo Complete! Check outputs\case3_rtx5090
echo.
pause
