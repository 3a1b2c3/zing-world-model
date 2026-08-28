@echo off
setlocal enabledelayedexpansion

echo ==========================================
echo Zing World Model - Example
echo ==========================================
echo.

cd /d "%~dp0"

if not exist ".venv" (
    echo ERROR: Virtual environment not found
    echo Run: setup_venv.bat
    exit /b 1
)

call .venv\Scripts\activate.bat
if errorlevel 1 exit /b 1

set PRETRAINED_DIR=%cd%\pretrained_models
set CHECKPOINT=%PRETRAINED_DIR%\checkpoint.pt
set MESSAGES=%cd%\examples\sample_input.jsonl
set OUTPUT_DIR=%cd%\outputs

if not exist "%PRETRAINED_DIR%" (
    echo ERROR: Models not found in %PRETRAINED_DIR%
    echo Run: download_models.bat
    exit /b 1
)

if not exist "%MESSAGES%" (
    echo ERROR: Example input not found: %MESSAGES%
    echo Create examples\sample_input.jsonl first
    exit /b 1
)

if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

set PYTHONPATH=%cd%\src;%PYTHONPATH%

echo Running inference...
echo   Pretrained: %PRETRAINED_DIR%
echo   Checkpoint: %CHECKPOINT%
echo   Input: %MESSAGES%
echo   Output: %OUTPUT_DIR%
echo.

python -m zing_v0_5 ^
  --pretrained-dir "%PRETRAINED_DIR%" ^
  --checkpoint "%CHECKPOINT%" ^
  --messages "%MESSAGES%" ^
  --output-dir "%OUTPUT_DIR%" ^
  --seed 42

if errorlevel 1 (
    echo ERROR: Inference failed
    exit /b 1
)

echo.
echo Complete!
echo Output: %OUTPUT_DIR%
echo.
