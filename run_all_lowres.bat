@echo off
setlocal enabledelayedexpansion

echo ==========================================
echo Zing-0.5 - All Low-Res Examples (352x640)
echo Load checkpoint ONCE, process all samples
echo ==========================================
echo.

cd /d "%~dp0"

call .venv\Scripts\activate.bat
set PYTHONPATH=%cd%\src;%PYTHONPATH%
set CUDA_VISIBLE_DEVICES=0

set PRETRAINED_DIR=%cd%\pretrained_models
set SNAPSHOTS_DIR=%PRETRAINED_DIR%\models--seedleap--zing-0.5\snapshots

for /d %%D in ("%SNAPSHOTS_DIR%\*") do (
    set SNAPSHOT_DIR=%%D
)

set PRETRAINED_PATH=%SNAPSHOT_DIR%\pretrained
set CHECKPOINT_PATH=%SNAPSHOT_DIR%\generator\model.pt

if exist "%OUTPUT_DIR%\all_lowres" (
    echo Skipping - outputs\all_lowres already exists
    echo.
    pause
    exit /b 0
)

echo Running all_lowres.jsonl (7 samples, 1 model load)...
echo.

python -m zing_v0_5 ^
  --pretrained-dir "%PRETRAINED_PATH%" ^
  --checkpoint "%CHECKPOINT_PATH%" ^
  --messages "examples\all_lowres.jsonl" ^
  --output-dir "outputs\all_lowres" ^
  --seed 0

if errorlevel 1 (
    echo.
    echo ERROR: Inference failed
    exit /b 1
)

echo.
echo Complete! Check outputs\all_lowres
echo.
pause
